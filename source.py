import json
import base64
from PIL import Image


# input json:
# {
#   "img": <BASE64-encoded img>,
#   "shape": [ {"x": <x>, "y": <y>}, {"x": <x>, "y": <y>}, {"x": <x>, "y": <y>}, {"x": <x>, "y": <y>} ]
# }
def parse_input_json(img_rect_json):
    img_rect = json.loads(img_rect_json)
    img = base64.b64decode(img_rect["img"])
    rect = img_rect["shape"]
    print("img:", img)
    print("rect:", rect)
    return img, rect


def save_background_in_image(img, object, bg_w, bg_h):
    left = round(img.width * object[0]['x'])
    top = round(img.height * object[0]['y']) - bg_h
    right = img.width - left - bg_w
    bottom = img.height - top - bg_h

    print(img.size, left, top, right, bottom)
    bg = img.crop((
        left,
        top,
        bg_w + left,
        bg_h + top))

    bg.save('background.png')

    #background.show()


test_json = open('test.json', 'r').read()
img_, object = parse_input_json(test_json)
filename = 'object.jpg'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(img_)
img = Image.open(filename)

object_width = round(img.width * (object[1]['x'] - object[0]['x']))
object_height = round(img.height * (object[2]['y'] - object[1]['y']))
print('object: height: ' + str(object_height) + " width: " + str(object_width))

background_width = round(object_width * 0.7)
background_height = round(object_height * 0.7)

print('background size: ', background_width, background_height)

save_background_in_image(img, object, background_width, background_height)
