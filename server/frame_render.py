import os
import cv2
import time
import json
import base64
import logging
import threading
import subprocess
import numpy as np
from aiohttp import web
from av import VideoFrame
from aiortc import VideoStreamTrack

# Tensorflow modules
import tensorflow as tf

# local modules
from backend.inpaint.inpaint import Inpaint
from backend.detection.detection import DetectionModel

# from object_detection.utils import ops, visualization_utils

try:
    cuda_visible_devices = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
                "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
                stdout=subprocess.PIPE).stdout.readlines()]))
    print('CUDA VISIBLE DEVICES:\n', cuda_visible_devices, '\n\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
except ValueError:
    logging.info('Get CUDA_VISIBLE_DEVICES: "0"\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

frames_time = {
    "boxes": [],
    "inpaint": [],
    "edges": []
}

CONFIG_PATH = './config.json'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)
    logging.info('Configuration project:\n' + str(json.dumps(CONFIG, sort_keys=True, indent=4)))


class Render:
    def __init__(self, inpaint_model=None, image_size=None, run_test=True, ):
        self.args = CONFIG["run_config"]
        self.warmup_iterations = self.args["num_warmup_iterations"]
        self.frame_size = tuple(image_size) if image_size else tuple(self.args['frame_size'])  # [<width>, <height>]

        logging.info('Get session and send test image')

        with tf.Graph().as_default() as tf_graph:
            # Connect to Tensorflow graph and run pretrained inpaint and object-detection models
            self.detection = DetectionModel(config=CONFIG, frame_size=self.frame_size, tf_graph=tf_graph)
            self.session = self.detection.session

            if self.args["inpaint"]:
                if inpaint_model is None:
                    # Load inpaint model
                    self.inpaint_model = Inpaint(
                        frame_size=self.frame_size,
                        reduction_ratio=self.args["reduction_ratio"],
                        session=self.session
                    )
                else:
                    # use given inpaint model
                    self.inpaint_model = inpaint_model

            if run_test:
                # Test run algorithm detection and inpainting (to overclock model)
                # Load test image
                img = cv2.resize(cv2.imread(self.args["test_image_inpaint_path"]), self.frame_size)
                # Test running
                self.run(img, transform=('inpaint' if self.args['inpaint'] else 'boxes'))

    def run(self, image, transform, objects_to_remove=None):
        """
        :param image: Initial image
        :param transform: Algorithm to use
        :param objects_to_remove: (optional) class objects to remove with "inpaint". Default: ["all"]
        :return: {
            "image": <image>, -- Result image
            "objects": <objects> -- All detected objects
        }
        """
        if not objects_to_remove:
            objects_to_remove = ['all']

        response = self.detection.detection(
            img=image,
            objects_to_remove=objects_to_remove,
            draw_box=transform == 'boxes'
        )

        if transform == 'inpaint' and response["objects"]:
            response = self.inpaint_model.inpaint_image(*response.values())

        img, objects = response["image"], response["objects"]

        return {
            "image": img,
            "objects": objects
        }


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform, render):
        super().__init__()  # don't forget this!
        self.track = track
        self.render = render
        self.transform = transform['name']
        self.args = CONFIG["run_config"]
        self.frame_size = tuple(self.args['frame_size'])  # [<width>, <height>]
        self.warmup_iterations = self.args["num_warmup_iterations"]
        self.objects = []
        # "all" - to remove all detected objects
        self.objects_to_remove = ["all"] if self.transform != 'inpaint' else transform['src']

    async def recv(self):
        start_time = time.time()
        frame = await self.track.recv()

        if self.transform and self.warmup_iterations <= 0:
            img = frame.to_ndarray(format="bgr24")

            if img.shape[:2] != self.frame_size:
                img = cv2.resize(img, self.frame_size)

            if self.transform == 'boxes' or self.transform == 'inpaint':
                res = self.render.run(image=img, transform=self.transform, objects_to_remove=self.objects_to_remove)
                img, self.objects = res["image"], res["objects"]
            else:
                # Perform edge detection
                # Default algorithm
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            render_time = (time.time() - start_time)
            frames_time[self.transform].append(render_time)
            if len(frames_time[self.transform]) > 10:
                del frames_time[self.transform][0]
            # logging.info('Return frame in %.5f sec' % render_time)
            return new_frame
        else:
            self.warmup_iterations -= 1
            log_average_time_interval()

        logging.info('Return warm up frame in %.5f sec' % (time.time() - start_time))
        return frame


def log_average_time_interval():
    threading.Timer(10.0, log_average_time_interval).start()
    logging.info('Average time: %s' % get_average_time_render('all'))


def get_average_time_render(algorithm):
    if algorithm == 'all':
        answer = {}
        for key, times in frames_time.items():
            answer[key] = np.mean(times)
        return answer

    elif algorithm in frames_time.keys():
        array = frames_time[algorithm].copy()
        average_time = np.mean(frames_time[algorithm]) if array else 1.0
        return average_time

    return None
