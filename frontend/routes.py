import numpy as np
import requests
import logging
import base64
import cv2
import json
import os
import uuid
import asyncio
from aiohttp import web
from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS

from werkzeug.contrib.fixers import ProxyFix

from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

# local modules
from backend import source


app = Flask(__name__)
CORS(app)

# URL = "http://127.0.0.1:5000/"
URL = "http://94.103.94.220:5000/"
loop = asyncio.get_event_loop()
pcs = set()


@app.route('/')
def init():
    logging.info('Run init page')
    return render_template("index.html")


@app.route('/test_masking')
def test_masking():
    test_json = json.loads(open('backend/test.json', 'r').read())
    imgs = make_api_request(url_server=URL, method_name='get_masking_image', img=test_json['img'],
                            objects=test_json['objects'], class_objects=test_json['class_objects'])
    return str(imgs)


@app.route('/test_inpaint')
def test_inpaint():
    test_json = json.loads(open('backend/test.json', 'r').read())
    imgs = make_api_request(url_server=URL, method_name='get_inpaint_image',
                            img=test_json['img'], objects=test_json['objects'])
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

        success, image = cv2.imencode('.png', image_np)
        encoded_image = base64.b64encode(image.tobytes())
        logging.info("Return Generate Masking Image")
        return make_api_response({'img': encoded_image.decode("utf-8")})

    except Exception as e:
        logging.error(e)
        return make_api_response({'message': 'Internal Server Error'}, code=500)


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()
        if self.transform == "inpaint":
            img = frame.to_ndarray(format="bgr24")
            objects, class_objects = source.test_objects()
            new_img = source.get_image_inpaint(img, objects)
            new_frame = VideoFrame.from_ndarray(new_img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        elif self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame


@app.route("/offer", methods=['POST'])
def offer():
    response = loop.run_until_complete(webrtc())
    return response


async def webrtc():
    logging.info('Request send' + str(request))
    params = request.get_json(force=True)
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    # pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    # logging.info(pc_id + " " + "Created for %s" % request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logging.info("ICE connection state is %s" % pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logging.info("Track %s received" % track.kind)

        local_video = VideoTransformTrack(track, transform=params["video_transform"])
        pc.addTrack(local_video)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logging.info('Return json')

    await asyncio.sleep(20)

    return make_response((jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}), 200))


def make_api_response(payload, code=200):
    return make_response((jsonify({'payload': payload}), code))


def make_api_request(url_server, method_name, **kwargs):
    # url = url_server + method_name
    url = url_server + method_name
    response = requests.post(url, json=kwargs).json()

    logging.debug(str(response))
    return response


app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run()
