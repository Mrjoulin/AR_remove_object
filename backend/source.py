import os
import cv2
import time
import base64
import random
import logging
import numpy as np
from PIL import Image
from backend.masking.generate_pattern import get_generative_background


def decode_input_image(image):
    try:
        decode_img = base64.b64decode(image.encode('utf-8'))
        path = 'backend/object.jpg'
        with open(path, 'wb') as write_file:
            write_file.write(decode_img)
        img = Image.open(path)
        os.remove(path)
    except:
        img = Image.fromarray(image)
    return img


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
        path = f'backend/masking/imgs/background/background_{str(number_object)}.png'
        bg.save(path)

        return {'success': True, 'bg_path': path}
    else:
        return {'success': False}


def get_image_masking(_img, objects, objects_class):
    # test_json = open('test.json', 'r').read()
    # _img, objects = parse_input_json(test_json)
    img = decode_input_image(_img)

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


def get_mask_objects(image, objects=None, masks=None, boxes=None, classes_to_render=None):
    mask_np = np.zeros(image.shape, np.uint8)

    if objects:
        for _object in objects:
            mask_np[int(_object['y']):int(_object['y'] + _object['height']),
                    int(_object['x']):int(_object['x'] + _object['width'])] = 255
    elif masks.any():
        mask_np = postprocess(mask_np, boxes, masks, draw=False, classes_to_render=classes_to_render)

    # start_time = time.time()
    # image_np = cv2.inpaint(image, mask_np, 0.1, cv2.INPAINT_NS)
    # logging.info('Inpaint image: %s sec' % (time.time() - start_time))
    # mask_np = np.expand_dims(mask_np, axis=2)
    return mask_np


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

        colorsFile = "AR_remover/objectdetection/colors.txt"
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


def remove_all_generate_files():
    pattern_path = 'backend/masking/pattern'
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
    img = Image.open('server/vk_bot/render_imgs/to_render_img_456243552.jpg')
    arr = np.array(img)
    objects, class_obj = test_objects()
    get_mask_objects(arr, objects=objects)
    logging.info("--- %s seconds ---" % (time.time() - start_time))


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
