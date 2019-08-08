import cv2
import time
import json
import base64
import logging
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


def connect_to_tensorflow_graph():
    PATH_TO_FROZEN_GRAPH = "./AR_remover/tensorflow-graph/fast_boxes/frozen_inference_graph.pb"
    PATH_TO_LABELS = './AR_remover/tensorflow-graph/mscoco_label_map.pbtxt'

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

    logging.info('Connecting Tensorflow model in %s sec' % (time.time() - render_time))
    return {
        'detection_graph': detection_graph,
        'category_index': category_index
    }


def object_detection(tf_gpaph,  img, session=None, inpaint=False):
    render_time = time.time()
    detection_graph = tf_gpaph['detection_graph']
    category_index = tf_gpaph['category_index']

    if session is None:
        sess = tf.Session(graph=detection_graph)
    else:
        sess = session

    logging.info('Start object detecting on image')
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

    out = sess.run(
        [num_detections, scores, boxes, classes],
        feed_dict={image_tensor: image_np_expanded})

    if not inpaint:
        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(out[2][0]),
            np.squeeze(out[3][0]).astype(np.int32),
            np.squeeze(out[1][0]),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

    num_detections = int(out[0][0])
    im_height, im_width = img.shape[:2]
    percent_detection = 0.4
    objects = []
    for i in range(num_detections):
        if out[1][0, i] > percent_detection:
            position = out[2][0][i]
            (xmin, xmax, ymin, ymax) = (
                position[1] * im_width, position[3] * im_width, position[0] * im_height,
                position[2] * im_height)

            width_object = xmax - xmin
            height_object = ymax - ymin
            objects.append({'x': int(xmin), 'y': int(ymin), 'width': width_object, 'height': height_object})

    logging.info(
        str([category_index.get(value) for index, value in enumerate(out[3][0])
             if out[1][0, index] > percent_detection])
    )

    logging.info('Render image in %s' % (time.time() - render_time))
    if session is not None:
        return {
            'objects': objects,
            'image': img
        }
    else:
        return sess


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        if self.transform:
            self.tf_graph = connect_to_tensorflow_graph()
            logging.info('Send test image and get session')
            img = cv2.imread('server/imgs/render_img.jpeg')
            self.session = object_detection(self.tf_graph, img)

            # Load inpaint model
            self.inpaint_model = Inpainting(session=self.session)
            output = self.inpaint_model.get_output(cv2.imread("server/imgs/inpaint.png"),
                                                   cv2.imread("server/imgs/mask_256.png"),
                                                   reuse=False)
            self.inpaint_model.load_model()
            self.inpaint_model.session.run(output)

    async def recv(self):
        start_time = time.time()
        frame = await self.track.recv()

        if self.transform:
            img = frame.to_ndarray(format="bgr24")

            if self.transform == 'boxes' or self.transform == 'inpaint':
                # Detection objects in frame
                response = object_detection(self.tf_graph, img, session=self.session,
                                            inpaint=self.transform == 'inpaint')
                img = response['image']
                objects = response['objects']

                if self.transform == 'inpaint':
                    # Remove needed objects buy inpaint algorithm
                    mask = source.get_mask_objects(img, objects=objects)
                    output = self.inpaint_model.get_output(img, mask, reuse=True)
                    # Inpaint image
                    result = self.inpaint_model.session.run(output)
                    img = result[0][:, :, ::-1]

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
            frame = VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = frame.pts
            frame.time_base = frame.time_base

        logging.info('Return frame in %s' % (time.time() - start_time))
        return frame
