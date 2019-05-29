import os
import cv2
import time
import base64
import logging
from io import BytesIO
import numpy as np
from PIL import Image
from backend.generation_pattern.generate_pattern import get_generative_background


def decode_input_image(image):
    try:
        decode_img = base64.b64decode(image.encode('utf-8'))
        image = Image.open(BytesIO(decode_img))
        path = 'backend/object.png'
        image.save(path)
        image_np = cv2.imread(path)
        os.remove(path)
    except:
        image_np = image
    logging.info("Image received")
    return image_np


def get_background_coordinates(img, bg_w, bg_h, current_obj, objects):
    flag = True
    if current_obj['y'] >= bg_h:
        for obj in objects:
            if obj['x'] + obj['width'] > current_obj['x'] or obj['x'] < current_obj['x'] + current_obj['width']:
                if obj['y'] + obj['height'] < current_obj['y']:
                    if current_obj['y'] - obj['y'] - obj['height'] < bg_h:
                        flag = False
                elif obj['y'] < current_obj['y']:
                    flag = False
        if flag:
            left = current_obj['x']
            top = current_obj['y'] - bg_h
            return left, top
    flag = True
    if img.width - current_obj['x'] - current_obj['width'] >= bg_w:
        for obj in objects:
            if obj['y'] + obj['height'] > current_obj['y'] or obj['y'] < current_obj['y'] + current_obj['height']:
                if obj['x'] > current_obj['x'] + current_obj['width']:
                    if obj['x'] - current_obj['x'] - current_obj['width'] < bg_w:
                        flag = False
                elif obj['x'] + obj['width'] > current_obj['x'] + current_obj['width']:
                    flag = False
        if flag:
            left = current_obj['x'] + current_obj['width']
            top = current_obj['y']
            return left, top
    flag = True
    if img.height - current_obj['y'] - current_obj['height'] >= bg_h:
        for obj in objects:
            if obj['x'] + obj['width'] > current_obj['x'] or obj['x'] < current_obj['x'] + current_obj['width']:
                if obj['y'] > current_obj['y'] + current_obj['height']:
                    if obj['y'] - current_obj['y'] - current_obj['height'] < bg_h:
                        flag = False
                elif obj['y'] + obj['height'] > current_obj['y'] + current_obj['height']:
                    flag = False
        if flag:
            left = current_obj['x']
            top = current_obj['y'] + current_obj['height']
            return left, top
    flag = True
    if current_obj['x'] >= bg_w:
        for obj in objects:
            if obj['y'] + obj['height'] > current_obj['y'] or obj['y'] < current_obj['y'] + current_obj['height']:
                if obj['x'] + obj['width'] < current_obj['x']:
                    if current_obj['x'] - obj['x'] - obj['width'] < bg_w:
                        flag = False
                elif obj['x'] < current_obj['x']:
                    flag = False
        if flag:
            left = current_obj['x'] - bg_w
            top = current_obj['y']
            return left, top
    return -1, -1


def save_background_image(img, objects, obj, number_object, bg_w, bg_h):
    left, top = get_background_coordinates(img, bg_w, bg_h, obj, objects)
    logging.info('image size: ' + str(img.size) + ' background coordinates: ' + str(left) + ' ' + str(top))
    if left != -1:
        bg = img.crop((
            left,
            top,
            bg_w + left,
            bg_h + top))
        path = f'backend/background/background_{str(number_object)}.png'
        bg.save(path)

        return {'success': True, 'bg_path': path}
    else:
        return {'success': False}


def get_image_masking(_img, objects, objects_class):
    # test_json = open('test.json', 'r').read()
    # _img, objects = parse_input_json(test_json)
    img = Image.fromarray(decode_input_image(_img))

    number_object = 0
    for current_object, object_class in zip(objects, objects_class):
        object_width = int(current_object['width'])
        object_height = int(current_object['height'])
        logging.info('object: height: {height} width: {width} class: {object_class}'.format(
            height=str(object_height), width=str(object_width), object_class=str(object_class)))

        # Crop background is 20 % of size object
        background_width = round(object_width * 0.25)
        background_height = round(object_height * 0.25)

        logging.info('background size: ' + str(background_width) + ' ' + str(background_height))

        request = save_background_image(img, objects, current_object,number_object, background_width, background_height)
        if request['success']:
            get_generative_background(request['bg_path'], object_width, object_height, object_class)

            backgrond = Image.open(f'backend/out/1/out_{str(object_class)}.jpg')
            img.paste(backgrond, (current_object['x'], current_object['y']))
            number_object += 1

    image_np = np.array(img)
    return image_np


def get_image_inpaint(_image, objects):
    image = decode_input_image(_image)

    image_np_mark = image.copy()
    mask_np = np.zeros(image_np_mark.shape[:2], np.uint8)
    for _object in objects:
        for x in range(int(_object['x']), int(_object['x'] + _object['width'])):
            for y in range(int(_object['y']), int(_object['y'] + _object['height'])):
                try:
                    mask_np[y][x] = 255
                except IndexError:
                    pass
    image_np = cv2.inpaint(image, mask_np, 1, cv2.INPAINT_TELEA)
    return image_np


def remove_all_generate_files():
    pattern_path = 'backend/generation_pattern/pattern'
    remove_pathes = [
        'backend/background/',
        'backend/out/1/',
    ]

    for path in remove_pathes:
        for img in os.listdir(path):
            os.remove(path + img)

    for form in ['.jpg', '.png']:
        if os.path.exists(pattern_path + form):
            os.remove(pattern_path + form)


def test_crop():
    # testing crop image
    start_time = time.time()
    img = Image.open('backend/test_img.jpg')
    arr = np.array(img)

    test_objects = [
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
    get_image_masking(arr, test_objects, class_obj)
    logging.info("--- %s seconds ---" % (time.time() - start_time))
