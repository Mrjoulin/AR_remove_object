import os
import sys
import cv2
import time
import json
import base64
import logging
import subprocess
import numpy as np
from PIL import Image
import tensorflow as tf
from distutils.version import StrictVersion


# Tensorflow object detection modules
from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

# local modules
from backend.source import *
from backend.feature.track_object import plane_tracker
from backend.inpaint.inpaint import Inpainting, NewInpainting
from backend.detection.trt_optimization.optimization import *
# from backend.detection.trt_detecton import inference # TRT/TF inference wrappers


try:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
                "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
                stdout=subprocess.PIPE).stdout.readlines()]))
except ValueError:
    logging.info('Get CUDA_VISIBLE_DEVICES: "0"')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

PATH_TO_CONFIG = './config.json'
PATH_TO_FROZEN_GRAPH = "./local/tensorflow-graph/fast_boxes/frozen_inference_graph.pb"
PATH_TO_LABELS = './local/tensorflow-graph/mscoco_label_map.pbtxt'


def camera(video_path=0):
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    start_time = time.time()
    while cap.isOpened():
        ret, image_np = cap.read()
        if ret:
            cnt += 1
            cv2.imshow('object detection', image_np)
        else:
            cap.release()
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

    logging.info('cnt %s' % cnt)
    logging.info('time %s' % (time.time() - start_time))
    logging.info('%s frames per second' % (cnt / (time.time() - start_time)))


def tensorflow_with_trt_render(cap, video_size):
    with open(PATH_TO_CONFIG, 'r') as f:
        config = json.load(f)
        logging.info('Configuration project:\n' + str(json.dumps(config, sort_keys=True, indent=4)))

    frozen_graph = build_model(
        **config['model_config']
    )

    # optimize model using source model
    frozen_graph = optimize_model(
        frozen_graph,
        **config['optimization_config']
    )

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    # Local variables
    args = config["run_config"]
    runtimes = []  # list of runtimes for each batch
    percent_detection = args["percent_detection"]
    num_warmup_iterations = args["num_warmup_iterations"]
    image_idx = 0
    inpaint = args["inpaint"]  # render_video or render_image
    small_size = (video_size[0] // args["reduction_ratio"], video_size[1] // args["reduction_ratio"])
    logging.info('Start rendering')
    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=sess_config) as sess:
            tf.import_graph_def(frozen_graph, name='')
            tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
            tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
            tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
            tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
            tf_num_detections = tf_graph.get_tensor_by_name(
                NUM_DETECTIONS_NAME + ':0')

            if inpaint:
                inpaint_session = Inpainting(session=sess)
                logging.info('Video shape: %s' % str(video_size))
                input_image_tf = tf.compat.v1.placeholder(
                    dtype=tf.float32,
                    shape=(1, small_size[1], small_size[0] * 2, 3)
                )
                output = inpaint_session.get_output(input_image_tf)
                inpaint_session.load_model()
                test_image = np.expand_dims(cv2.resize(cv2.imread(args["test_image_inpaint_path"]), small_size), 0)
                test_mask = np.expand_dims(cv2.resize(cv2.imread(args["test_mask_inpaint_path"]), small_size), 0)

                test_input_image = np.concatenate([test_image, test_mask], axis=2)
                inpaint_session.session.run(output, feed_dict={input_image_tf: test_input_image})

            while cap.isOpened():
                ret, image_np = cap.read()
                start_time = time.time()
                if ret:
                    initial_image = image_np.copy()
                    image_np = cv2.resize(image_np, small_size if inpaint else video_size)
                    logging.info('Render image shape: %s' % str(image_np.shape))
                else:
                    cap.release()
                    cv2.destroyAllWindows()
                    break

                batch_images = np.expand_dims(image_np, axis=0)

                # run num_warmup_iterations outside of timing
                if image_idx < num_warmup_iterations:
                    boxes, classes, scores, num_detections = sess.run(
                        [tf_boxes, tf_classes, tf_scores, tf_num_detections],
                        feed_dict={tf_input: batch_images})
                    image_idx += 1
                else:
                    # execute model and compute time difference
                    t0 = time.time()
                    boxes, classes, scores, num_detections = sess.run(
                        [tf_boxes, tf_classes, tf_scores, tf_num_detections],
                        feed_dict={tf_input: batch_images})
                    t1 = time.time()

                    # log runtime and image count
                    runtimes.append(float(t1 - t0))
                    logging.info("Detection time: %.4f ms" % (runtimes[-1] * 1000))

                objects = []
                for i in range(int(num_detections)):
                    if scores[0, i] > percent_detection:
                        position = boxes[0][i]
                        (xmin, xmax, ymin, ymax) = (position[1], position[3], position[0], position[2])

                        objects.append({'position': {'x_min': xmin, 'y_min': ymin, 'x_max': xmax, 'y_max': ymax}})

                        class_id = int(classes[0][i])
                        score = float(scores[0][i])
                        logging.info('classId: %s; score: %s; box: %s' % (class_id, score, position))

                        if not inpaint:
                            # Visualization of the results of a detection.
                            color = (136, 218, 43)
                            cv2.rectangle(image_np, (int(xmin * video_size[0]), int(ymin * video_size[1])),
                                          (int(xmax * video_size[0]), int(ymax * video_size[1])), color, 8)

                if objects:
                    # get inpainting image
                    if inpaint:
                        # Get mask objects
                        mask_np = get_mask_objects(image_np, objects)
                        # Inpainting Image
                        frame_time = time.time()
                        image_np = image_np * (1 - mask_np) + 255 * mask_np
                        image_np = np.expand_dims(image_np, 0)
                        input_mask = np.expand_dims(255 * mask_np, 0)
                        input_image = np.concatenate([image_np, input_mask], axis=2)
                        result = inpaint_session.session.run(output, feed_dict={input_image_tf: input_image})
                        image_np = merge_inpaint_image_to_initial(initial_image, mask_np, result[0][:, :, ::-1])
                        logging.info('Frame time: %s sec' % (time.time() - frame_time))

                logging.info('---- %s ms ----' % ((time.time() - start_time) * 1000))
                cv2.imshow('object detection with TensorRT', image_np)

                if len(runtimes) % 10 == 0:
                    logging.info('Average time: %s' % np.mean(runtimes))

                if cv2.waitKey(10) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break


def tensorflow_render(cap, video_size, render_image=False, render_video=False, number_video=None, tf2=False):
    frame_per_second = 30.0
    if render_video:
        inpaint_name = "videos/out_inpaint_video%s%s.mp4" % (('_%s' % number_video) if number_video else '',
                                                             '_tf2' if tf2 else '')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_inpaint_video = cv2.VideoWriter(inpaint_name, fourcc, frame_per_second, video_size, True)

    # Creating detection graph and config to Tensorflow Session
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # Local variables
    inpaint = False  # render_video or render_image
    small_size = (video_size[0] // 2, video_size[1] // 2)
    procent_detecion = 0.5
    frames_time = []
    first_frame = True
    logging.info('Start rendering')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=sess_config) as sess:
            if inpaint:
                # Load inpaint model
                inpaint_session = Inpainting(session=sess)
                logging.info('Video shape: %s' % str(video_size))
                input_image_tf = tf.placeholder(dtype=tf.float32, shape=(1, small_size[1], small_size[0] * 2, 3))
                output = inpaint_session.get_output(input_image_tf)
                inpaint_session.load_model()
                test_image = np.expand_dims(cv2.resize(cv2.imread("server/imgs/inpaint.png"), small_size), 0)
                test_mask = np.expand_dims(cv2.resize(cv2.imread("server/imgs/mask.png"), small_size), 0)
                test_input_image = np.concatenate([test_image, test_mask], axis=2)
                inpaint_session.session.run(output, feed_dict={input_image_tf: test_input_image})

                # Load new inpaint model
                # inpaint_session = NewInpainting(session=sess)
                # image = cv2.imread("server/imgs/inpaint_480.png")
                # mask = cv2.imread("server/imgs/mask_480.png", 0).astype(np.float32)
                # mask = np.expand_dims(mask, axis=2) / 255
                # input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, image.shape[0], image.shape[1], 3])
                # input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, image.shape[0], image.shape[1], 1])
                # output = inpaint_session.get_output(input_image_tf, input_mask_tf, reuse=False)
                # inpaint_session.load_model()
                # image = image * (1 - mask) + 255 * mask
                # image = np.expand_dims(image, 0)
                # mask = np.expand_dims(mask, 0)
                # result = inpaint_session.session.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
                # cv2.imwrite('server/imgs/output_480.png', result[0][:, :, ::-1])

            while cap.isOpened():
                start_time = time.time()
                ret, image_np = cap.read()
                if ret:
                    initial_image = image_np.copy()
                    image_np = cv2.resize(image_np, small_size if inpaint else video_size)
                    logging.info('Render image shape: %s' % str(image_np.shape))
                else:
                    cap.release()
                    if render_video:
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
                # if not inpaint:
                #     vis_util.visualize_boxes_and_labels_on_image_array(
                #        image_np,
                #        np.squeeze(out[2][0]),
                #        np.squeeze(out[3][0]).astype(np.int32),
                #        np.squeeze(out[1][0]),
                #        category_index,
                #        use_normalized_coordinates=True,
                #        line_thickness=8)

                num_detections = int(out[0][0])
                im_height, im_width = image_np.shape[:2]
                objects = []
                # objects_class = []
                for i in range(num_detections):
                    if out[1][0, i] > procent_detecion:
                        position = out[2][0][i]
                        (xmin, xmax, ymin, ymax) = (position[1], position[3], position[0], position[2])

                        objects.append({'position': {'x_min': xmin, 'y_min': ymin, 'x_max': xmax, 'y_max': ymax}})

                        class_id = int(out[3][0][i])
                        score = float(out[1][0][i])
                        logging.info('classId: %s; score: %s; box: %s' % (class_id, score, position))

                        # Visualization of the results of a detection.
                        if not inpaint:
                            color = (136, 218, 43)
                            cv2.rectangle(image_np, (int(xmin*im_width), int(ymin*im_height)),
                                          (int(xmax*im_width), int(ymax*im_height)), color, 8)

                # Visualize detected bounding boxes.
                logging.info(
                    str([category_index.get(value) for index, value in enumerate(out[3][0]) if out[1][0, index] > 0.5])
                )
                logging.info('Detection objects in %s sec' % (time.time() - start_time))

                # if Tensorflow find objects
                if objects:
                    # get inpainting image
                    if inpaint:
                        # Get mask objects
                        mask_np = get_mask_objects(image_np, objects)
                        # Inpainting Image
                        frame_time = time.time()
                        image_np = image_np * (1 - mask_np) + 255 * mask_np
                        image_np = np.expand_dims(image_np, 0)
                        input_mask = np.expand_dims(255 * mask_np, 0)
                        input_image = np.concatenate([image_np, input_mask], axis=2)
                        result = inpaint_session.session.run(output, feed_dict={input_image_tf: input_image})
                        image_np = merge_inpaint_image_to_initial(initial_image, mask_np, result[0][:, :, ::-1])
                        logging.info('Frame time: %s sec' % (time.time() - frame_time))

                if render_image:
                    path_to_save = 'local/imgs/render_inpaint_image.png'
                    logging.info('Save image to %s' % path_to_save)
                    success, image = cv2.imencode('.png', image_np)
                    with open(path_to_save, 'wb') as save_file:
                        save_file.write(image.tobytes())

                cv2.imshow('object detection', cv2.resize(image_np, video_size))

                if render_video:
                    logging.info('Write a inpaint render moment')
                    out_inpaint_video.write(image_np)

                def get_screen(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDBLCLK:
                        screens = os.listdir('backend/screens')
                        screens.sort()
                        number_screen = screens[-1].split('_')[1].split('.')[0]
                        logging.info('\n\n    Save screen with number %s\n' % number_screen)
                        cv2.imwrite('backend/screens/screenshot_%s.png' % (int(number_screen) + 1), image_np)

                cv2.setMouseCallback('object detection', get_screen)

                render_time = time.time() - start_time
                logging.info("--- %s seconds ------- " % render_time)
                if not first_frame:                
                    frames_time.append(render_time)
                    logging.info("Average time %s sec" % (sum(frames_time) / len(frames_time)))
                else:
                    first_frame = False

                if cv2.waitKey(20) & 0xFF == ord(' '):
                    inpaint = not inpaint
                    logging.info('Start inpainting objects' if inpaint else 'Stop inpaint')

                if cv2.waitKey(1) & 0xFF == ord('p'):
                    logging.info('P is pressed')
                    plane_tracker.App(0).run()
                    break

                if cv2.waitKey(15) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break
