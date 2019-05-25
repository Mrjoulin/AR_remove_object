import requests
import logging
import base64
import json
from flask import Flask, request, jsonify, make_response, render_template
from werkzeug.contrib.fixers import ProxyFix

# local modules
from backend.source import get_image_background_fragment


app = Flask(__name__)


@app.route('/')
def init():
    logging.info('Run init page')
    return render_template("index.html")


@app.route('/test')
def test():
    test_json = json.loads(open('backend/test.json', 'r').read())
    imgs = make_api_request('get_pattern', img=test_json['img'], objects=test_json['objects'],
                            class_objects=test_json['class_objects'])
    return str(imgs)


# input json:
# {
#   "img": <BASE64-encoded img>,
#   "objects": [ {"x": <x>, "y": <y>, "width": <width>, "height": <height>}, ...]
#   "class_objects": [<number_class>, ...]
# }
#
# output json:
# {
#   "payload": {
#       <class_object>: <BASE64-encoded pattern object>
#       ...
# }
# }


@app.route("/get_pattern", methods=['POST'])
def get_pattern():
    if request.is_json:
        json = request.get_json()
        logging.info('Json received')
    else:
        logging.error('BAD REQUEST JSON')
        return make_api_response({'message': 'No Content'}, code=204)

    if 'img' in json and isinstance(json['img'], str) and \
            'objects' in json and isinstance(json['objects'], list) and \
            'class_objects' in json and isinstance(json['class_objects'], list) and \
            len(json['objects']) == len(json['class_objects']):
        img = json['img']
        objects = json['objects']
        class_objects = json['class_objects']
    else:
        return make_api_response({'message': 'Partial Content'}, code=206)

    get_image_background_fragment(img, objects, class_objects)

    patterns = {}
    for class_object in class_objects:
        try:
            with open('backend/out/1/out_%s.jpg' % str(class_object), 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read())
                patterns[str(class_object)] = encoded_string.decode("utf-8")
        except FileNotFoundError:
            return make_api_response({'message': 'Internal Server Error'}, code=500)

    return make_api_response(patterns)


def make_api_response(payload, code=200):
    return make_response((jsonify({'payload': payload}), code))


def make_api_request(method_name, **kwargs):
    # url = "http://localhost:5000/" + method_name
    url = "http://94.103.94.220:5000/" + method_name
    response = requests.post(url, json=kwargs).json()

    logging.debug(str(response))
    return response


app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run()
