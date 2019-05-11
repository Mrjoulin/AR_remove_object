import numpy as np
import sys
import tensorflow as tf
import cv2
import time
import logging
from distutils.version import StrictVersion
from PIL import Image

# Tensorflow object detection modules
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# local modules
from backend import source
from backend.track_object import plane_tracker

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


def find_object_in_image():
    cap = cv2.VideoCapture(0)
    PATH_TO_FROZEN_GRAPH = "./AR_remover/frozen_inference_graph.pb"
    PATH_TO_LABELS = './AR_remover/mscoco_label_map.pbtxt'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    render = False
    class_to_hide = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                start_time = time.time()
                ret, image_np = cap.read()
                ret, initial_image = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
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
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(out[2][0]),
                    np.squeeze(out[3][0]).astype(np.int32),
                    np.squeeze(out[1][0]),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                num_detections = int(out[0][0])
                im_height, im_width = image_np.shape[:2]
                objects = []
                objects_class = []
                for i in range(num_detections):
                    if out[1][0, i] > 0.4:
                        position = out[2][0][i]
                        (xmin, xmax, ymin, ymax) = (
                            position[1] * im_width, position[3] * im_width, position[0] * im_height,
                            position[2] * im_height)

                        width_object = xmax - xmin
                        height_object = ymax - ymin
                        objects.append({'x': int(xmin), 'y': int(ymin), 'width': width_object, 'height': height_object})

                # Visualize detected bounding boxes.
                logging.info(
                    str([category_index.get(value) for index, value in enumerate(out[3][0]) if out[1][0, index] > 0.5])
                )

                # for i in range(num_detections):
                for index, value in enumerate(out[3][0]):
                    if out[1][0, index] > 0.4:
                        class_id = int(out[3][0][index])
                        score = float(out[1][0][index])
                        box = [float(v) for v in out[2][0][index]]
                        objects_class.append(class_id)

                        logging.info('classId ' + str(class_id) + ' score ' + str(score) + ' box ' + str(box))

                        if class_id in class_to_hide:
                            image = Image.fromarray(image_np)
                            backgrond = Image.open(f'backend/out/1/out_{str(class_id)}.jpg')
                            left = round(im_width * box[1]) - 5
                            top = round(im_height * box[0]) - 14
                            resize_width = round(im_width * (box[3] - box[1])) + 10
                            resize_height = round(im_height * (box[2] - box[0])) + 28
                            normal_bg = backgrond.resize((resize_width, resize_height), Image.ANTIALIAS)
                            image.paste(normal_bg, (left, top))
                            image_np = np.array(image)

                cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

                def get_screen(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDBLCLK:
                        screen = Image.fromarray(image_np)
                        screen.save('backend/out/screens/screenshot.png')

                cv2.setMouseCallback('object detection', get_screen)

                if cv2.waitKey(1) & 0xFF == ord(' '):
                    source.get_image_background_fragment(initial_image, objects, objects_class)
                    for index, value in enumerate(out[3][0]):
                        if out[1][0, index] > 0.4:
                            class_id = int(out[3][0][index])
                            class_to_hide.append(class_id)
                    render = True

                if cv2.waitKey(20) & 0xFF == ord('p'):
                    logging.info('P is pressed')
                    plane_tracker.App(0).run()
                    break

                if cv2.waitKey(25) & 0xFF == ord('c'):
                    logging.info('Clear mask')
                    render = False
                    class_to_hide = []

                if cv2.waitKey(30) & 0xFF == ord('q') or cv2.waitKey(31) == 27:
                    cv2.destroyAllWindows()
                    break

                logging.info("--- %s seconds ---" % (time.time() - start_time))
