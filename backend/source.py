import json
import base64
from PIL import Image


# input json:
# {
#   "img": <BASE64-encoded img>,
#   "shapes": [ {"x": <x>, "y": <y>, "width": <width>, "height": <height>}, ...]
# }
def parse_input_json(img_rect_json):
    img_rect = json.loads(img_rect_json)
    img = base64.b64decode(img_rect["img"])
    rects = img_rect["shapes"]
    print("img:", img)
    print("rect:", rects)
    return img, rects


def get_background_coordinates(img, bg_w, bg_h, current_object, objects):
    flag = True
    if current_object['y'] >= bg_h:
        for object in objects:
            if object['x'] + object['width'] > current_object['x'] or object['x'] < current_object['x'] + current_object['width']:
                if object['y'] + object['height'] < current_object['y']:
                    if current_object['y'] - object['y'] - object['height'] < bg_h:
                        flag = False
                elif object['y'] < current_object['y']:
                    flag = False
        if flag:
            left = current_object['x']
            top = current_object['y'] - bg_h
            return left, top
    flag = True
    if img.width - current_object['x'] - current_object['width'] >= bg_w:
        for object in objects:
            if object['y'] + object['height'] > current_object['y'] or object['y'] < current_object['y'] + current_object['height']:
                if object['x'] > current_object['x'] + current_object['width']:
                    if object['x'] - current_object['x'] - current_object['width'] < bg_w:
                        flag = False
                elif object['x'] + object['width'] > current_object['x'] + current_object['width']:
                    flag = False
        if flag:
            left = current_object['x'] + current_object['width']
            top = current_object['y']
            return left, top
    flag = True
    if img.height - current_object['y'] - current_object['height'] >= bg_h:
        for object in objects:
            if object['x'] + object['width'] > current_object['x'] or object['x'] < current_object['x'] + current_object['width']:
                if object['y'] > current_object['y'] + current_object['height']:
                    if object['y'] - current_object['y'] - current_object['height'] < bg_h:
                        flag = False
                elif object['y'] + object['height'] > current_object['y'] + current_object['height']:
                    flag = False
        if flag:
            left = current_object['x']
            top = current_object['y'] + current_object['height']
            return left, top
    flag = True
    if current_object['x'] >= bg_w:
        for object in objects:
            if object['y'] + object['height'] > current_object['y'] or object['y'] < current_object['y'] + current_object['height']:
                if object['x'] + object['width'] < current_object['x']:
                    if current_object['x'] - object['x'] - object['width'] < bg_w:
                        flag = False
                elif object['x'] < current_object['x']:
                    flag = False
        if flag:
            left = current_object['x'] - bg_w
            top = current_object['y']
            return left, top
    return 0, 0


def save_background_in_image(img, objects, object, number_object, bg_w, bg_h):
    left, top = get_background_coordinates(img, bg_w, bg_h, object, objects)
    print('image size:', img.size, 'background coordinates:', left, top)
    bg = img.crop((
        left,
        top,
        bg_w + left,
        bg_h + top))

    bg.save(f'background/background_{str(number_object)}.png')
    return bg


def post_background_fragment(bg_fragment):
    pass


test_json = open('test.json', 'r').read()
img_, objects = parse_input_json(test_json)
filename = 'object.jpg'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(img_)
img = Image.open(filename)

number_object = 0
for object in objects:
    object_width = object['width']
    object_height = object['height']
    print('object: height: ' + str(object_height) + " width: " + str(object_width))

    background_width = round(object_width * 0.25)
    background_height = round(object_height * 0.25)

    print('background size: ', background_width, background_height)

    background_fragment = save_background_in_image(img, objects, object, number_object, background_width, background_height)
    number_object += 1
