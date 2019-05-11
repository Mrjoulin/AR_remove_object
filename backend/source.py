import json
import time
import base64
import logging
import numpy as np
from PIL import Image
from backend.generation_pattern.generate_pattern import get_generative_background


def parse_input_json(img_rect_json):
    img_rect = json.loads(img_rect_json)
    img = base64.b64decode(img_rect["img"])
    rects = img_rect["shapes"]
    logging.info("img:" + str(img))
    logging.info("rect:" + str(rects))
    return img, rects


def get_background_line_coordinates(img, bg_w, bg_h, current_obj, objects):
    flag = True
    if current_obj['y'] >= bg_h:
        for obj in objects:
            if obj['x'] + obj['width'] > current_obj['x'] or obj['x'] < current_obj['x'] + \
                    current_obj['width']:
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
            if obj['y'] + obj['height'] > current_obj['y'] or obj['y'] < current_obj['y'] + \
                    current_obj['height']:
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
            if obj['x'] + obj['width'] > current_obj['x'] or obj['x'] < current_obj['x'] + \
                    current_obj['width']:
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
            if obj['y'] + obj['height'] > current_obj['y'] or obj['y'] < current_obj['y'] + \
                    current_obj['height']:
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


def save_background_image(img, objects, obj, object_class, number_object, bg_w, bg_h):
    left, top = get_background_coordinates(img, bg_w, bg_h, obj, objects)
    logging.info('image size: ' + str(img.size) + ' background coordinates: ' + str(left) + ' ' + str(top))
    if left != -1:
        bg = img.crop((
            left,
            top,
            bg_w + left,
            bg_h + top))
        path = f'backend/background/background_{str(number_object)}.png'

        generate_background = True
    else:
        bg = Image.open('backend/background/grid_background.jpg')
        path = f'backend/out/1/out_{str(object_class)}.jpg'

        generate_background = False

    bg.save(path)

    return generate_background, path


def get_image_background_fragment(_img, objects, objects_class):
    # test_json = open('test.json', 'r').read()
    # _img, objects = parse_input_json(test_json)
    try:
        img_arr = base64.b64decode(_img)
        filename = 'backend/object.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(img_arr)
        img = Image.open(filename)
    except OSError:
        img = Image.fromarray(_img)

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

        gen_bg, path = save_background_image(img, objects, current_object, object_class,
                                             number_object, background_width, background_height)
        if gen_bg:
            get_generative_background(path, object_width, object_height, object_class)

        number_object += 1


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
    get_image_background_fragment(arr, test_objects, class_obj)
    logging.info("--- %s seconds ---" % (time.time() - start_time))
