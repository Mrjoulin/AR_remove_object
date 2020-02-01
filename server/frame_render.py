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
from backend import source
from backend.inpaint.inpaint import Inpainting
from backend.detection.trt_optimization.optimization import *

from object_detection.utils import ops, visualization_utils

try:
    cuda_visible_devices = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
                "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
                stdout=subprocess.PIPE).stdout.readlines()]))
    print('CUDA VISIBLE DEVICES:\n', cuda_visible_devices, '\n\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
except ValueError:
    logging.info('Get CUDA_VISIBLE_DEVICES: "0"')
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
    def __init__(self):
        self.args = CONFIG["run_config"]
        self.frame_size = tuple(self.args['frame_size'])  # [<width>, <height>]
        self.warmup_iterations = self.args["num_warmup_iterations"]

        # Connect to Tensorflow graph and run pretrained inpaint and object-detection models
        frozen_graph = self.connect_to_tensorflow_graph()

        logging.info('Get session and send test image')

        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(frozen_graph, name='')
            self.tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
            tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
            tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
            tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
            tf_num_detections = tf_graph.get_tensor_by_name(NUM_DETECTIONS_NAME + ':0')
            self.tf_params = [tf_boxes, tf_classes, tf_scores, tf_num_detections]
            if self.args['use_masks_objects']:
                tf_masks = tf_graph.get_tensor_by_name(MASKS_NAME + ':0')
                detection_boxes = tf.squeeze(tf_boxes, [0])
                detection_masks = tf.squeeze(tf_masks, [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tf_num_detections[0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, self.frame_size[0], self.frame_size[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tf_masks = tf.expand_dims(detection_masks_reframed, 0)
                self.tf_params.append(tf_masks)

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.session = tf.Session(config=sess_config)

            if self.args["inpaint"]:
                # Load inpaint model
                self.inpaint_model = Inpainting(session=self.session)

                small_width = self.frame_size[0] // self.args["reduction_ratio"]
                small_height = self.frame_size[1] // self.args["reduction_ratio"]
                self.small_size = (
                    small_width - small_width % 8,
                    small_height - small_height % 8
                )
                self.input_image_tf = tf.placeholder(
                    dtype=tf.float32,
                    shape=(1, self.small_size[1], self.small_size[0] * 2, 3)
                )
                self.output = self.inpaint_model.get_output(self.input_image_tf)
                self.inpaint_model.load_model()
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

        response = self.object_detection(image, objects_to_remove, draw_box=transform == 'boxes')

        if transform == 'inpaint' and response["objects"]:
            response = self.inpaint_image(*response.values())

        img, objects = response["image"], response["objects"]

        return {
            "image": img,
            "objects": objects
        }

    def connect_to_tensorflow_graph(self):
        render_time = time.time()

        frozen_graph = build_model(
            **CONFIG['model_config']
        )

        # optimize model using source model
        frozen_graph = optimize_model(
            frozen_graph,
            **CONFIG['optimization_config']
        )
        logging.info('Optimized Tensorflow model in %.5f sec' % (time.time() - render_time))
        return frozen_graph

    def object_detection(self, img, objects_to_remove, draw_box=False):
        image_np_expanded = np.expand_dims(img, axis=0)

        # Actual detection.
        if self.args['use_masks_objects']:
            boxes, classes, scores, num_detections, masks = self.session.run(
                self.tf_params,
                feed_dict={self.tf_input: image_np_expanded})
            logging.info(str(masks[0]))
            logging.info(str(masks[0].shape))
        else:
            boxes, classes, scores, num_detections = self.session.run(
                self.tf_params,
                feed_dict={self.tf_input: image_np_expanded})
        percent_detection = self.args["percent_detection"]
        objects = []
        for i in range(int(num_detections)):
            if scores[0, i] > percent_detection:
                class_id = int(classes[0][i])

                if ('all' in objects_to_remove) or (class_id in objects_to_remove):
                    position = boxes[0][i]
                    (xmin, xmax, ymin, ymax) = (position[1], position[3], position[0], position[2])

                    objects.append(
                        {
                            "class_id": class_id,
                            "position": {
                                'x_min': xmin,
                                'y_min': ymin,
                                'x_max': xmax,
                                'y_max': ymax
                            }
                        }
                    )

                    mask = None
                    if self.args['use_masks_objects']:
                        mask = masks[0][i]
                        mask = (mask > percent_detection).astype(np.uint8)
                        objects[-1]["mask"] = mask

                    if draw_box:
                        color = (136, 218, 43)  # RGB
                        if mask is not None:
                            alph = 0.5
                            for j in range(3):
                                img = img * (1 - mask * (1 - alph)) + alph * mask * color[j]
                        else:
                            cv2.rectangle(img, (int(xmin * self.frame_size[0]), int(ymin * self.frame_size[1])),
                                          (int(xmax * self.frame_size[0]), int(ymax * self.frame_size[1])), color, 8)

        return {
            'objects': objects,
            'image': img
        }

    def inpaint_image(self, img, objects):
        init_img = img.copy()
        img = cv2.resize(img, self.small_size)
        # Remove needed objects by inpaint algorithm
        frame_time = time.time()
        # Get mask objects
        mask, objects = source.get_mask_objects(img, objects=objects)
        # Inpaint image
        img = img * (1 - mask) + 255 * mask
        img = np.expand_dims(img, 0)
        input_mask = np.expand_dims(255 * mask, 0)
        input_image = np.concatenate([img, input_mask], axis=2)
        result = self.inpaint_model.session.run(self.output, feed_dict={self.input_image_tf: input_image})
        # Merge the result of program to initial image
        img = source.merge_inpaint_image_to_initial(init_img, mask, result[0][:, :, ::-1])
        return {
            'objects': objects,
            'image': img
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
        for key in frames_time.keys():
            answer[key] = np.mean(frames_time[key])
        return answer

    elif algorithm in frames_time.keys():
        array = frames_time[algorithm].copy()
        average_time = np.mean(frames_time[algorithm]) if array else 1.0
        return average_time

    return None
