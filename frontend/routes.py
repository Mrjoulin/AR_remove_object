from PIL import Image
import numpy as np
import requests
import logging
import base64
import json
import os
from flask import Flask, request, jsonify, make_response, render_template
from werkzeug.contrib.fixers import ProxyFix

# local modules
from backend import source


app = Flask(__name__)


@app.route('/')
def init():
    logging.info('Run init page')
    return render_template("index.html")


@app.route('/test_masking')
def test_masking():
    test_json = json.loads(open('backend/test.json', 'r').read())
    imgs = make_api_request('get_masking_image', img=test_json['img'], objects=test_json['objects'],
                            class_objects=test_json['class_objects'])
    return str(imgs)


@app.route('/test_inpaint')
def test_inpaint():
    test_json = json.loads(open('backend/test.json', 'r').read())
    imgs = make_api_request('get_inpaint_image', img=test_json['img'], objects=test_json['objects'])
    return str(imgs)


@app.route("/get_masking_image", methods=['POST'])
def get_masking_image():

    # ------------- GET MASKING IMAGE -------------
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
    #       "img": <BASE64-encoded masking image>
    # }
    # }

    return get_image(masking=True)

    #patterns = {}
    #for class_object in class_objects:
    #    try:
    #        with open('backend/out/1/out_%s.jpg' % str(class_object), 'rb') as image_file:
    #            encoded_string = base64.b64encode(image_file.read())
    #            patterns[str(class_object)] = encoded_string.decode("utf-8")
    #    except FileNotFoundError:
    #        return make_api_response({'message': 'Internal Server Error'}, code=500)


@app.route('/get_inpaint_image', methods=['POST'])
def get_inpaint_image():

    # ------------- GET INPAINT IMAGE -------------
    # input json:
    # {
    #   "img": <BASE64-encoded img>,
    #   "objects": [ {"x": <x>, "y": <y>, "width": <width>, "height": <height>}, ...]
    # }
    #
    # output json:
    # {
    #   "payload": {
    #       "img": <BASE64-encoded inpaint image>
    # }
    # }

    return get_image(inpaint=True)


def get_image(masking=False, inpaint=False):
    if request.is_json:
        json = request.get_json()
        logging.info('Json received')
    else:
        logging.error('BAD REQUEST JSON')
        return make_api_response({'message': 'No Content'}, code=204)

    if 'img' in json and isinstance(json['img'], str) and \
            'objects' in json and isinstance(json['objects'], list):
        img = json['img']
        objects = json['objects']
        if masking and 'class_objects' in json and isinstance(json['class_objects'], list) and \
            len(json['objects']) == len(json['class_objects']):
            class_objects = json['class_objects']
    else:
        return make_api_response({'message': 'Partial Content'}, code=206)

    try:
        if masking:
            image_np = source.get_image_masking(img, objects, class_objects)
        elif inpaint:
            image_np = source.get_image_inpaint(img, objects)
        else:
            image_np = np.array(source.decode_input_image(img))

        source.remove_all_generate_files()

        image = Image.fromarray(image_np)
        path_img = 'backend/object.jpg'
        image.save(path_img)
        with open(path_img, 'rb') as file:
            encoded_image = base64.b64encode(file.read())
        os.remove(path_img)
        logging.info("Return Generate Masking Image")
        return make_api_response({'img': encoded_image.decode("utf-8")})

    except Exception as e:
        logging.error(e)
        return make_api_response({'message': 'Internal Server Error'}, code=500)


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
