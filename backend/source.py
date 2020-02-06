import os
import cv2
import base64
import random
import logging

# Local modules
from backend.detection.trt_detecton.coco import *


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


def get_mask_objects(image, objects):
    mask_np = np.zeros(image.shape, np.uint8)
    h, w = image.shape[:2]

    for obj in objects:
        if 'mask' in obj:
            mask_np = mask_np + cv2.resize(obj.pop('mask'), (w, h)).reshape((h, w, 1))
        else:
            mask_np[int(obj['position']['y_min'] * h):int(obj['position']['y_max'] * h),
                    int(obj['position']['x_min'] * w):int(obj['position']['x_max'] * w)] = 1

    return mask_np, objects


def merge_inpaint_image_to_initial(initial_image, inpaint_mask, inpaint_image):
    initial_size = (initial_image.shape[1], initial_image.shape[0])
    big_mask = cv2.resize(inpaint_mask, initial_size)
    inpaint_objects = cv2.resize(inpaint_image, initial_size) * big_mask
    image_np = initial_image * (1 - big_mask)
    return image_np + inpaint_objects


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
