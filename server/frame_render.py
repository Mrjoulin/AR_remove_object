import os
import cv2
import time
import json
import base64
import logging
import subprocess
import numpy as np
from aiohttp import web
from av import VideoFrame
from aiortc import VideoStreamTrack

# Tensorflow modules
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# local modules
from backend import source
from backend.inpaint.inpaint import Inpainting

os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
            "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
            stdout=subprocess.PIPE).stdout.readlines()]))
frames_time = []


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.first_frame = True
        if self.transform == 'boxes' or self.transform == 'inpaint':
            self.tf_graph = self.connect_to_tensorflow_graph()
            logging.info('Get session and send test image')

            with self.tf_graph['detection_graph'].as_default():
                sess_config = tf.ConfigProto()
                sess_config.gpu_options.allow_growth = True
                self.session = tf.Session(graph=self.tf_graph['detection_graph'], config=sess_config)
            
                # Load inpaint model
                if self.transform == 'inpaint':
                    self.inpaint_model = Inpainting(session=self.session)
                    video_size = (240, 320)
                    self.input_image_tf = tf.placeholder(dtype=tf.float32, shape=(1, video_size[0], video_size[1]*2, 3))
                    self.output = self.inpaint_model.get_output(self.input_image_tf)
                    self.inpaint_model.load_model()
                    # Test run algorithm (to overclock model)
                    test_image = np.expand_dims(cv2.imread("server/imgs/inpaint_%s.png" % video_size[0]), 0)
                    test_mask = np.expand_dims(cv2.imread("server/imgs/mask_%s.png" % video_size[0]), 0)
                    input_image = np.concatenate([test_image, test_mask], axis=2)
                    self.inpaint_model.session.run(self.output, feed_dict={self.input_image_tf: input_image})

                # Test run algorithm detection (to overclock model)
                img = cv2.imread('server/imgs/render_img.jpeg')
                self.object_detection(img)

    async def recv(self):
        start_time = time.time()
        frame = await self.track.recv()

        if self.transform and not self.first_frame:
            img = frame.to_ndarray(format="bgr24")

            if self.transform == 'boxes' or self.transform == 'inpaint':
                # Detection objects in frame
                init_img = img.copy()
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                response = self.object_detection(img, draw_box=self.transform != 'inpaint')
                img = response['image']
                objects = response['objects']

                if self.transform == 'inpaint':
                    # Remove needed objects buy inpaint algorithm
                    frame_time = time.time()
                    # Get mask objects
                    mask = source.get_mask_objects(img, objects=objects)
                    # Inpaint image
                    img = img * (1 - mask) + 255 * mask
                    img = np.expand_dims(img, 0)
                    input_mask = np.expand_dims(255 * mask, 0)
                    input_image = np.concatenate([img, input_mask], axis=2)
                    result = self.inpaint_model.session.run(self.output, feed_dict={self.input_image_tf: input_image})
                    img = source.merge_inpaint_image_to_initial(init_img, mask, result[0][:, :, ::-1])
                    logging.info('Frame time: %.5f sec' % (time.time() - frame_time))

            elif self.transform == "edges":
                # perform edge detection
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            elif self.transform == "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(cv2.medianBlur(img_edges, 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                  cv2.THRESH_BINARY, 9, 2)
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            render_time = (time.time() - start_time)
            frames_time.append(render_time)
            logging.info('Return frame in %.5f sec' % render_time)
            return new_frame

        self.first_frame = False
        logging.info('Return frame in %.5f sec' % (time.time() - start_time))
        return frame

    def connect_to_tensorflow_graph(self):
        PATH_TO_FROZEN_GRAPH = "./AR_remover/tensorflow-graph/fast_boxes/frozen_inference_graph.pb"
        PATH_TO_LABELS = './AR_remover/tensorflow-graph/mscoco_label_map.pbtxt'

        logging.info('Using Tensorflow Detection Graph and labels:\n%s\n%s' % (PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS))

        render_time = time.time()
        # net = cv2.dnn.readNetFromTensorflow(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        logging.info('Connecting Tensorflow model in %.5f sec' % (time.time() - render_time))
        return {
            'detection_graph': detection_graph,
            'category_index': category_index
        }

    def object_detection(self, img, draw_box=False):
        render_time = time.time()
        detection_graph = self.tf_graph['detection_graph']
        category_index = self.tf_graph['category_index']

        with detection_graph.as_default():
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(img, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.

            out = self.session.run(
                [num_detections, scores, boxes, classes],
                feed_dict={image_tensor: image_np_expanded})

            if draw_box:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    np.squeeze(out[2][0]),
                    np.squeeze(out[3][0]).astype(np.int32),
                    np.squeeze(out[1][0]),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

            num_detections = int(out[0][0])
            percent_detection = 0.5
            objects = []
            for i in range(num_detections):
                if out[1][0, i] > percent_detection:
                    position = out[2][0][i]
                    (xmin, xmax, ymin, ymax) = (position[1], position[3], position[0], position[2])

                    objects.append({'x_min': xmin, 'y_min': ymin, 'x_max': xmax, 'y_max': ymax})

            logging.info(
                'Number detected objects: ' +
                str(len([category_index.get(value) for index, value in enumerate(out[3][0])
                         if out[1][0, index] > percent_detection]))
            )

            logging.info('Detection object in frame: %.5f sec' % (time.time() - render_time))
            return {
                'objects': objects,
                'image': img
            }


def get_average_time_render():
    return sum(frames_time) / (len(frames_time) if frames_time else 1)


def get_image(request_json, masking=False, inpaint=False):
    if 'img' in request_json and isinstance(request_json['img'], str) and \
            'objects' in request_json and isinstance(request_json['objects'], list):
        img = request_json['img']
        objects = request_json['objects']
        if masking and 'class_objects' in request_json and isinstance(request_json['class_objects'], list) and \
                len(request_json['objects']) == len(request_json['class_objects']):
            class_objects = request_json['class_objects']
    else:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'message': 'Partial Content'}
            ),
        )

    try:
        if masking:
            image_np = source.get_image_masking(img, objects, class_objects)
        elif inpaint:
            image_np = source.get_mask_objects(img, objects)
        else:
            image_np = np.array(source.decode_input_image(img))

        source.remove_all_generate_files()

        success, image = cv2.imencode('.png', image_np)
        encoded_image = base64.b64encode(image.tobytes())
        logging.info("Return Generate Masking Image")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'img': encoded_image.decode("utf-8")}
            ),
        )

    except Exception as e:
        logging.error(e)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'message': 'Internal Server Error'}
            ),
        )
