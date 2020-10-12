import logging
import time

from backend.detection.trt_optimization.optimization import *
from object_detection.utils import ops, visualization_utils


class DetectionModel:
    def __init__(self, config, frame_size, tf_graph=None, session=None):
        self.config = config
        self.args = config["run_config"]
        self.frame_size = frame_size  # [<width>, <height>]
        self.normalize_objects = self.args['normalize_objects']
        self.normalize_size_limit = self.args["normalize_size_limit"]

        frozen_graph = self.connect_to_tensorflow_graph()

        if tf_graph is None:
            tf_graph = tf.Graph().as_default()

        tf.import_graph_def(frozen_graph, name='')
        self.tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
        tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
        tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
        tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
        tf_num_detections = tf_graph.get_tensor_by_name(NUM_DETECTIONS_NAME + ':0')
        self.tf_params = [tf_boxes, tf_classes, tf_scores, tf_num_detections]
        if self.args['use_masks_objects']:
            self.add_masks(tf_graph=tf_graph, tf_boxes=tf_boxes, tf_num_detections=tf_num_detections)

        if not session:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            session = tf.Session(config=sess_config)

        self.session = session

    def connect_to_tensorflow_graph(self):
        render_time = time.time()

        if self.config['use_trt']:
            frozen_graph = build_model(
                **self.config['model_config']
            )
            # optimize model using source model
            frozen_graph = optimize_model(
                frozen_graph,
                **self.config['optimization_config']
            )
        else:
            frozen_graph = tf.GraphDef()
            path_fo_frozen_graph = os.path.join(
                self.config["model_config"]["input_dir"],
                MODELS[self.config["model_config"]["model_name"]].extract_dir,
                "frozen_inference_graph.pb"
            )
            with tf.gfile.GFile(path_fo_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                frozen_graph.ParseFromString(serialized_graph)

        logging.info('Get Tensorflow model in %.5f sec' % (time.time() - render_time))
        return frozen_graph

    def add_masks(self, tf_graph, tf_boxes, tf_num_detections):
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

    def detection(self, img, objects_to_remove, draw_box=False):
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
        logging.info('Detected %s objects. Scores: %s. Classes: %s' % (
            int(num_detections),
            scores[0][:int(num_detections)],
            classes[0][:int(num_detections)]
        ))
        for i in range(int(num_detections)):
            if scores[0, i] > percent_detection:
                class_id = int(classes[0][i])

                if ('all' in objects_to_remove) or (class_id in objects_to_remove):
                    position = boxes[0][i]
                    (xmin, xmax, ymin, ymax) = (position[1], position[3], position[0], position[2])

                    h, w = ymax - ymin, xmax - xmin

                    if w <= self.normalize_size_limit:
                        logging.info('Normalize object width (class %s, object index %s)' % (class_id, len(objects)))
                        xmin = max(xmin - self.normalize_objects * w, 0)
                        xmax = min(xmax + self.normalize_objects * w, 1)

                    if h <= self.normalize_size_limit:
                        logging.info('Normalize object height (class %s, object index %s)' % (class_id, len(objects)))
                        ymin = max(ymin - self.normalize_objects * h, 0)
                        ymax = min(ymax + self.normalize_objects * h, 1)

                    objects.append(
                        {
                            "class_id": class_id,
                            "position": {
                                'x_min': float(xmin),
                                'y_min': float(ymin),
                                'x_max': float(xmax),
                                'y_max': float(ymax)
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
                                          (int(xmax * self.frame_size[0]), int(ymax * self.frame_size[1])), color, 4)

        return {
            'image': img,
            'objects': objects
        }
