import numpy as np
import sys
import os
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


def camera(video_path=0):
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    start_time = time.time()
    while cap.isOpened():
        ret, image_np = cap.read()
        try:
            cnt += 1
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
        except Exception as e:
            logging.debug(e)
            break
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

    logging.info('cnt %s' % cnt)
    logging.info('time %s' % (time.time() - start_time))
    logging.info('%s frames per second' % (cnt / (time.time() - start_time)))


def find_object_in_image(cap, video_size, render_image_by_masking=False, render_video_by_masking=False,
                         render_image_by_inpainting=False, render_video_by_inpainting=False, number_video=None):
    if render_video_by_masking:
        masking_name = 'videos/out_videos/out_masking_video%s.mp4' % (('_%s' % number_video) if number_video else '')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_masking_video = cv2.VideoWriter(masking_name, fourcc, 30.0, video_size, True)

    if render_video_by_inpainting:
        inpaint_name = 'videos/out_videos/out_inpaint_video%s.mp4' % (('_%s' % number_video) if number_video else '')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_inpaint_video = cv2.VideoWriter(inpaint_name, fourcc, 30.0, video_size, True)

    PATH_TO_FROZEN_GRAPH = "./AR_remover/tensorflow-graph/frozen_inference_graph2.pb"
    PATH_TO_LABELS = './AR_remover/tensorflow-graph/mscoco_label_map.pbtxt'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    render = render_video_by_masking
    inpaint = render_video_by_inpainting
    # class_to_hide = []
    procent_detecion = 0.4
    logging.info('Start rendering')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while cap.isOpened():
                start_time = time.time()
                ret, image_np = cap.read()
                if ret:
                    initial_image = image_np.copy()
                else:
                    cap.release()
                    if render_video_by_masking:
                        logging.info('Extracting masking render video in %s' % masking_name)
                        out_masking_video.release()
                    if render_video_by_inpainting:
                        logging.info('Extracting inpainting render video in %s' % inpaint_name)
                        out_inpaint_video.release()
                    cv2.destroyAllWindows()
                    break
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
                    if out[1][0, i] > procent_detecion:
                        position = out[2][0][i]
                        (xmin, xmax, ymin, ymax) = (
                            position[1] * im_width, position[3] * im_width, position[0] * im_height,
                            position[2] * im_height)

                        width_object = xmax - xmin
                        height_object = ymax - ymin
                        objects.append({'x': int(xmin), 'y': int(ymin), 'width': width_object, 'height': height_object})

                # Visualize detected bounding boxes.
                logging.info(
                    str([category_index.get(value) for index, value in enumerate(out[3][0]) if out[1][0, index] > 0.4])
                )

                # update classes objects to hide
                for index, value in enumerate(out[3][0]):
                    if out[1][0, index] > procent_detecion:
                        class_id = int(out[3][0][index])
                        score = float(out[1][0][index])
                        box = [float(v) for v in out[2][0][index]]

                        logging.info('classId ' + str(class_id) + ' score ' + str(score) + ' box ' + str(box))

                        objects_class.append(class_id)

                        # if class_id in class_to_hide:
                        #     image = Image.fromarray(image_np)
                        #     backgrond = Image.open(f'backend/out/1/out_{str(class_id)}.jpg')
                        #     left = round(im_width * box[1]) - 5
                        #     top = round(im_height * box[0]) - 14
                        #     resize_width = round(im_width * (box[3] - box[1])) + 10
                        #     resize_height = round(im_height * (box[2] - box[0])) + 28
                        #     normal_bg = backgrond.resize((resize_width, resize_height), Image.ANTIALIAS)
                        #     image.paste(normal_bg, (left, top))
                        #     image_np = np.array(image)

                # get masking image
                if render:
                    image_np = source.get_image_masking(initial_image, objects, objects_class)

                    # for class_img in class_img_to_hide:
                    #    class_to_hide.append(class_img)
                    # render = render_video_by_masking

                # get inpainting image
                if inpaint:
                    image_np = source.get_image_inpaint(initial_image, objects)

                if render_image_by_masking or render_image_by_inpainting:
                    path_to_save = 'AR_remover/render_' + ('masking' if render_image_by_masking else 'inpaint') + \
                                   '_image.jpeg'
                    logging.info('Save image to %s' % path_to_save)
                    img = Image.fromarray(image_np)
                    img.save(path_to_save)

                cv2.imshow('object detection', cv2.resize(image_np, video_size))

                if render_video_by_masking:
                    logging.info('Write a masking render moment')
                    out_masking_video.write(image_np)

                if render_video_by_inpainting:
                    logging.info('Write a inpaint render moment')
                    out_inpaint_video.write(image_np)

                def get_screen(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDBLCLK:
                        screen = Image.fromarray(image_np)
                        screens = os.listdir('backend/out/screens')
                        screens.sort()
                        number_screen = screens[-1][12:(13 if len(screens) < 10 else 14)]
                        screen.save('backend/out/screens/screenshot_%s.png' % (int(number_screen) + 1))

                cv2.setMouseCallback('object detection', get_screen)

                if cv2.waitKey(10) & 0xFF == ord(' '):
                    render = not render
                    logging.info('Start masking object' if render else 'Clear mask')
                    # class_to_hide = []

                if cv2.waitKey(20) & 0xFF == ord('i'):
                    inpaint = not inpaint
                    logging.info('Start inpainting objects' if inpaint else 'Stop inpaint')

                if cv2.waitKey(1) & 0xFF == ord('p'):
                    logging.info('P is pressed')
                    plane_tracker.App(0).run()
                    break

                if cv2.waitKey(15) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break

                logging.info("--- %s seconds ---" % (time.time() - start_time))

    source.remove_all_generate_files()
