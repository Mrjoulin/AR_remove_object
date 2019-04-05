import json
import base64


# input json:
# {
#   "img": <BASE64-encoded img>,
#   "rect": [ {"x": <x>, "y": <y>}, {"x": <x>, "y": <y>}, {"x": <x>, "y": <y>}, {"x": <x>, "y": <y>} ]
# }
def get_tiled_image(img_rect_json):
    img_rect = json.loads(img_rect_json)
    img = base64.b64decode(img_rect["img"])
    rect = img_rect["rect"]
    print("img:", img)
    print("rect:", rect)
    return 0


test_json = open('test.json', 'r').read()
get_tiled_image(test_json)



