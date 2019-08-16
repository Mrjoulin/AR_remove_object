import os
import sys
import cv2
import time
import base64
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from distutils.version import StrictVersion


# Tensorflow object detection modules
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# local modules
from backend import source
from backend.feature.track_object import plane_tracker
from backend.inpaint.inpaint import Inpainting, NewInpainting
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


def opencv_render(cap, video_size, use_server=False, render_image_by_masking=False, render_video_by_masking=False,
                  render_image_by_inpainting=False, render_video_by_inpainting=False, number_video=None):
    if render_video_by_masking:
        masking_name = f"videos/out_videos/out_google_cv_masking_video" \
                       f"{('_%s' % number_video) if number_video else ''}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_masking_video = cv2.VideoWriter(masking_name, fourcc, 30.0, video_size, True)

    if render_video_by_inpainting:
        inpaint_name = f"videos/out_videos/out_google_cv_inpaint_video" \
                       f"{('_%s' % number_video) if number_video else ''}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_inpaint_video = cv2.VideoWriter(inpaint_name, fourcc, 30.0, video_size, True)

    PATH_TO_FROZEN_GRAPH = "./AR_remover/tensorflow-graph/fast_boxes/frozen_inference_graph.pb"
    PATH_TO_LABELS = './AR_remover/tensorflow-graph/fast_boxes/mscoco_label_map.pbtxt'

    net = cv2.dnn.readNetFromTensorflow(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load the network

    render = render_video_by_masking or render_image_by_masking
    inpaint = render_video_by_inpainting or render_image_by_inpainting
    # class_to_hide = []
    procent_detecion = 0.5
    logging.info('Start rendering')
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

        # Create a 4D blob from a frame.
        #blob = cv2.dnn.blobFromImage(image_np, swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(image_np, swapRB=True, crop=False)
        # Set the input to the network
        net.setInput(blob)

        # Run the forward pass to get output from the output layers
        boxes_time = time.time()
        boxes = net.forward()
        logging.info('Boxes render in %s' % (time.time() - boxes_time))

        # out_mask = net.forward()
        # logging.info(str(out_mask))
        # logging.info(str(out_mask.flatten()))
        # boxes = net.forward(['detection_out_final'])
        # logging.info(boxes)
        # masks = net.forward(['detection_masks'])
        # logging.info(masks)

        # Extract the bounding box and mask for each of the detected objects
        if inpaint:
            image_np = source.get_mask_objects(image_np, boxes=boxes)
        else:
            # logging.info('Find boxes %s' % boxes)
            # classes_id = source.postprocess(image_np, boxes, get_class_to_render=True)
            # logging.info('Find Classes Id : %s' % classes_id)
            # image_np = source.postprocess(image_np, boxes, draw=True)

            rows, cols, channels = image_np.shape
            classes = ["background", "person", "bicycle", "car", "motorcycle",
                       "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                       "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                       "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
                       "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
                       "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                       "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
                       "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                       "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
                       "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
                       "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
                       "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))

            for detection in boxes[0,0,:,:]:
                score = float(detection[2])
                if score > procent_detecion:
                    left = detection[3] * cols
                    top = detection[4] * rows
                    right = detection[5] * cols
                    bottom = detection[6] * rows

                    # draw a red rectangle around detected objects
                    cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

                    idx = int(detection[1])

                    label = "{}: {:.2f}%".format(classes[idx], score * 100)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(image_np, label, (int(left), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        cv2.imshow('object detection', cv2.resize(image_np, video_size))

        if render_video_by_masking:
            logging.info('Write a masking render moment')
            out_masking_video.write(image_np)

        if render_video_by_inpainting:
            logging.info('Write a inpaint render moment')
            out_inpaint_video.write(image_np)

        def get_screen(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                screens = os.listdir('backend/out/screens')
                screens.sort()
                number_screen = screens[-1].split('_')[1].split('.')[0]
                logging.info('last number %s' % number_screen)
                cv2.imwrite('backend/masking/imgs/out/screens/screenshot_%s.png' % (int(number_screen) + 1), image_np)

        cv2.setMouseCallback('object detection', get_screen)

        if cv2.waitKey(10) & 0xFF == ord(' '):
            render = not render
            logging.info('Start masking object' if render else 'Clear mask')
            # class_to_hide = []

        if cv2.waitKey(20) & 0xFF == ord('i'):
            inpaint = not inpaint
            logging.info('Start inpainting objects' if inpaint else 'Stop inpaint')

        if cv2.waitKey(5) & 0xFF == ord('s'):
            use_server = not use_server
            logging.info('Start using server to rendering image' if use_server else 'Stop using server')

        if cv2.waitKey(1) & 0xFF == ord('p'):
            logging.info('P is pressed')
            plane_tracker.App(0).run()
            break

        if cv2.waitKey(15) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

        logging.info("--- %s seconds ---" % (time.time() - start_time))


def tensorflow_render(cap, video_size, use_server=False, render_image_by_masking=False, render_video_by_masking=False,
                      render_image_by_inpainting=False, render_video_by_inpainting=False, number_video=None, tf2=False):

    frame_per_second = 30.0
    if render_video_by_masking:
        masking_name = f"videos/out_videos/out_masking_video{('_%s' % number_video) if number_video else ''}" \
                       f"{'_tf2' if tf2 else ''}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_masking_video = cv2.VideoWriter(masking_name, fourcc, frame_per_second, video_size, True)

    if render_video_by_inpainting:
        inpaint_name = f"videos/out_videos/out_inpaint_video{('_%s' % number_video) if number_video else ''}" \
                       f"{'_tf2' if tf2 else ''}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_inpaint_video = cv2.VideoWriter(inpaint_name, fourcc, frame_per_second, video_size, True)

    PATH_TO_FROZEN_GRAPH = "./AR_remover/tensorflow-graph/frozen_inference_graph%s.pb" % ('2' if tf2 else '')
    PATH_TO_LABELS = './AR_remover/tensorflow-graph/mscoco_label_map.pbtxt'

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
    render = render_video_by_masking or render_image_by_masking
    inpaint = True  # render_video_by_inpainting or render_image_by_inpainting
    procent_detecion = 0.4
    render_frames = 0
    logging.info('Start rendering')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=sess_config) as sess:
            # Load inpaint model
            inpaint_session = Inpainting(session=sess)
            logging.info('Video shape: %s' % str(video_size))
            input_image_tf = tf.placeholder(dtype=tf.float32, shape=(1, video_size[1], video_size[0] * 2, 3))
            output = inpaint_session.get_output(input_image_tf)
            inpaint_session.load_model()
            test_image = np.expand_dims(cv2.imread("server/imgs/inpaint_480.png"), 0)
            test_mask = np.expand_dims(cv2.imread("server/imgs/mask_480.png"), 0)
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

                # if Tensorflow find objects
                if objects:
                    # get masking image
                    if render:
                        image_np = source.get_image_masking(initial_image, objects, objects_class)

                        # for class_img in class_img_to_hide:
                        #    class_to_hide.append(class_img)
                        # render = render_video_by_masking

                    # get inpainting image
                    if inpaint:
                        # Get mask objects
                        mask_np = source.get_mask_objects(initial_image, objects)
                        # Inpainting Image
                        frame_time = time.time()
                        # initial_image = initial_image * (1 - mask_np) + 255 * mask_np
                        initial_image = np.expand_dims(initial_image, 0)
                        mask_np = np.expand_dims(mask_np, 0)
                        input_image = np.concatenate([initial_image, mask_np], axis=2)
                        result = inpaint_session.session.run(output, feed_dict={input_image_tf: input_image})
                        image_np = result[0][:, :, ::-1]
                        logging.info('Frame time: %s sec' % (time.time() - frame_time))

                if render_image_by_masking or render_image_by_inpainting:
                    path_to_save = 'AR_remover/imgs/render_' + ('masking' if render_image_by_masking else 'inpaint') + \
                                   '_image.png'
                    logging.info('Save image to %s' % path_to_save)
                    success, image = cv2.imencode('.png', image_np)
                    with open(path_to_save, 'wb') as save_file:
                        save_file.write(image.tobytes())

                cv2.imshow('object detection', cv2.resize(image_np, video_size))

                if render_video_by_masking:
                    logging.info('Write a masking render moment')
                    out_masking_video.write(image_np)

                if render_video_by_inpainting:
                    logging.info('Write a inpaint render moment')
                    out_inpaint_video.write(image_np)

                def get_screen(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDBLCLK:
                        screens = os.listdir('backend/out/screens')
                        screens.sort()
                        number_screen = screens[-1].split('_')[1].split('.')[0]
                        logging.info('last number %s' % number_screen)
                        cv2.imwrite('backend/masking/imgs/out/screens/screenshot_%s.png' % (int(number_screen) + 1),
                                    image_np)

                cv2.setMouseCallback('object detection', get_screen)

                logging.info("--- %s seconds ---" % (time.time() - start_time))

                if cv2.waitKey(10) & 0xFF == ord(' '):
                    render = not render
                    logging.info('Start masking object' if render else 'Clear mask')
                    # class_to_hide = []

                if cv2.waitKey(20) & 0xFF == ord('i'):
                    inpaint = not inpaint
                    logging.info('Start inpainting objects' if inpaint else 'Stop inpaint')

                if cv2.waitKey(5) & 0xFF == ord('s'):
                    use_server = not use_server
                    logging.info('Start using server to rendering image' if use_server else 'Stop using server')

                if cv2.waitKey(1) & 0xFF == ord('p'):
                    logging.info('P is pressed')
                    plane_tracker.App(0).run()
                    break

                if cv2.waitKey(15) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break

                if render_video_by_inpainting or render_video_by_masking:
                    render_frames += 1
                    logging.info("Rendering %s seconds video" % (render_frames / frame_per_second))

    source.remove_all_generate_files()
