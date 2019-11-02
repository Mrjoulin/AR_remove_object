import os
import cv2
import base64
import random
import logging

# Local modules
from backend.detection.trt_detecton.coco import *

# Layout of TensorRT network output metadata
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

VISUALIZATION_THRESHOLD = 0.5


def decode_input_image(image):
    try:
        decode_img = base64.b64decode(image.encode('utf-8'))
        path = 'backend/object.jpg'
        with open(path, 'wb') as write_file:
            write_file.write(decode_img)
        image = cv2.imread(path)
        os.remove(path)
    except:
        pass
    return image


def get_mask_objects(image, objects=None, masks=None, boxes=None, classes_to_render=None):
    mask_np = np.zeros(image.shape, np.uint8)
    h, w = image.shape[:2]

    if objects:
        for obj in objects:
            mask_np[int(obj['position']['y_min'] * h):int(obj['position']['y_max'] * h),
                    int(obj['position']['x_min'] * w):int(obj['position']['x_max'] * w)] = 1
    elif masks.any():
        mask_np = postprocess(mask_np, boxes, masks, draw=False, classes_to_render=classes_to_render)

    return mask_np


def merge_inpaint_image_to_initial(initial_image, inpaint_mask, inpaint_image):
    initial_size = (initial_image.shape[1], initial_image.shape[0])
    big_mask = cv2.resize(inpaint_mask, initial_size)
    inpaint_objects = cv2.resize(inpaint_image, initial_size) * big_mask
    image_np = initial_image * (1 - big_mask)
    return image_np + inpaint_objects


def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    The prediction fields layout is described in TRT_PREDICTION_LAYOUT.

    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.

    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out

    Returns:
        Prediction field corresponding to given data.
    """
    shift = TRT_PREDICTION_LAYOUT[field_name]
    return detection_out[pred_start_idx + shift]


def analyze_prediction(detection_out, pred_start_idx, image_np):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    if confidence > VISUALIZATION_THRESHOLD:
        class_name = COCO_CLASSES_LIST[label]
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))

        im_height, im_width = image_np.shape[:2]
        print(image_np.shape)
        print('xmin {}, xmax {}, ymin {}, ymax {}'.format(xmin, xmax, ymin, ymax))

        cv2.rectangle(image_np, (int(xmin * im_width), int(ymin * im_height)),
                      (int(xmax * im_width), int(ymax * im_height)), (192, 168, 15), 6)

    return image_np


def postprocess(frame, boxes, masks=None, draw=False, get_class_to_render=False, classes_to_render=None):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    confThreshold = 0.4
    maskThreshold = 0.1
    if masks is not None:
        numClasses = masks.shape[1]
    numDetections = boxes.shape[2]
    logging.info('Number Detections %s' % numDetections)

    frameH = frame.shape[0]
    frameW = frame.shape[1]

    if get_class_to_render:
        classIds = []
        for i in range(numDetections):
            box = boxes[0, 0, i]
            score = box[2]
            if score > confThreshold:
                classId = int(box[1])
                if (classId + 1) not in classIds:
                    classIds.append(classId + 1)
        return classIds

    for i in range(numDetections):
        box = boxes[0, 0, i]
        if masks is not None:
            mask = masks[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])

            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            if masks is not None:
                # Extract the mask for the object
                classMask = mask[classId]
            else:
                classMask = None

            if draw:
                # Draw bounding box, colorize and show the mask on the image
                return drawBox(frame, classId, score, left, top, right, bottom, classMask, maskThreshold)
            elif classMask is not None and (classes_to_render is None or
                                            (classes_to_render is not None and (classId + 1) in classes_to_render)):
                classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
                mask = (classMask > maskThreshold)
                frame[top:bottom + 1, left:right + 1] = mask.astype(np.uint8)

    return frame


def drawBox(frame, classId, conf, left, top, right, bottom, classMask, maskThreshold):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    if classMask is not None:
        # Resize the mask, threshold, color and apply it on the image
        classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
        mask = (classMask > maskThreshold)
        roi = frame[top:bottom + 1, left:right + 1][mask]

        colorsFile = "local/objectdetection/colors.txt"
        with open(colorsFile, 'rt') as f:
            colorsStr = f.read().rstrip('\n').split('\n')
        colors = []
        for i in range(len(colorsStr)):
            rgb = colorsStr[i].split(' ')
            color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
            colors.append(color)

        # color = colors[classId % len(colors)]
        # Comment the above line and uncomment the two lines below to generate different instance colors
        colorIndex = random.randint(0, len(colors)-1)
        color = colors[colorIndex]

        # frame[top:bottom + 1, left:right + 1][mask] = \
        #     ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

        # Draw the contours on the image
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)

    return frame


def test_objects():
    objects = [
        {
            'x': 20,
            'y': 50,
            'width': 100,
            'height': 40
        },
        {
            'x': 350,
            'y': 20,
            'width': 100,
            'height': 40
        },
        {
            'x': 380,
            'y': 130,
            'width': 100,
            'height': 40
        },
        {
            'x': 130,
            'y': 135,
            'width': 100,
            'height': 40
        }
    ]
    class_obj = [1001, 1002, 1003, 1004]
    return objects, class_obj
