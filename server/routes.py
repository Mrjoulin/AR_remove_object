import os
import cv2
import ssl
import json
import uuid
import time
import base64
import asyncio
import logging
import requests
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

# local modules
from backend import source
from server.frame_render import VideoTransformTrack, get_image

URL = "http://127.0.0.1:5000/"
# URL = "http://84.201.133.73:5000/"
ROOT = os.path.dirname(os.path.abspath(__file__))
pcs = set()


async def init(request):
    logging.info('Run init page from IP: %s' % request.remote)
    content = open(os.path.join(ROOT, "templates/true_thanos_web/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def init_css(request):
    content = open(os.path.join(ROOT, "templates/true_thanos_web/static/css/style.css"), "r").read()
    return web.Response(content_type="text/css", text=content)


async def init_js(request):
    content = open(os.path.join(ROOT, "templates/thanosar/js/webRTC.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def test_masking(request):
    test_json = json.loads(open('backend/test.json', 'r').read())
    imgs = make_api_request(url_server=URL, method_name='get_masking_image', img=test_json['img'],
                            objects=test_json['objects'], class_objects=test_json['class_objects'])
    return web.Response(text=str(imgs))


async def test_inpaint(requset):
    test_json = json.loads(open('backend/test.json', 'r').read())
    imgs = make_api_request(url_server=URL, method_name='get_inpaint_image',
                            img=test_json['img'], objects=test_json['objects'])
    response = await imgs.json()
    logging.info('Return respose')
    return web.Response(text=str(response))


async def get_masking_image(request):

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
    try:
        request_json = await request.json()
        logging.info('Json received')
    except:
        logging.error('BAD REQUEST JSON')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'message': 'No Content'}
            ),
        )

    return get_image(request_json, masking=True)

    # patterns = {}
    # for class_object in class_objects:
    #    try:
    #        with open('backend/out/1/out_%s.jpg' % str(class_object), 'rb') as image_file:
    #            encoded_string = base64.b64encode(image_file.read())
    #            patterns[str(class_object)] = encoded_string.decode("utf-8")
    #    except FileNotFoundError:
    #        return make_api_response({'message': 'Internal Server Error'}, code=500)


async def get_inpaint_image(request):

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
    logging.info('Get request')
    try:
        request_json = await request.json()
        logging.info('Json received')
    except:
        logging.error('BAD REQUEST JSON')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'message': 'No Content'}
            ),
        )

    return get_image(request_json, inpaint=True)


async def offer(request):
    '''
    :param request:
    sdp, type: <string>, <string> - for WebRTC connection
    video_transform: {
                        name: <name_algorithm> - Options: "boxes", "inpaint", "edges", "cartoon" or empty "".
                        src: [<additional variables>] - for "inpaint" -- [<class object>] (For example: ['people'])
                                                        for others -- []
    :return:
        "boxes" - stream frames with visualized detected objects
        "inpaint" - stream frames with remove selected object
    '''

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    logging.info(pc_id + " " + "Created for %s" % request.remote)

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
        logging.info("Track {kind} received. Video transform: {transform}".format(kind=track.kind,
                                                                                  transform=params["video_transform"]))

        local_video = VideoTransformTrack(track, transform=params["video_transform"])
        pc.addTrack(local_video)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        headers={
            "Access-Control-Allow-Origin": "*"
        },
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


def make_api_request(url_server, method_name, **kwargs):
    # url = url_server + method_name
    url = url_server + method_name
    logging.info('Send post in %s' % url)
    response = requests.post(url, json=kwargs).json()
    logging.info('Get response %s' % response)
    logging.debug(str(response))
    return response


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def run_app(port=5000, host=None, cert_file=None, key_file=None):
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', init)
    app.router.add_get('/static/css/style.css', init_css)
    # app.router.add_get('/js/webRTC.js', init_js)
    app.router.add_get('/test_masking', test_masking)
    app.router.add_get('/test_inpaint', test_inpaint)
    app.router.add_post('/get_masking_image', get_masking_image)
    app.router.add_post('/get_inpaint_image', get_inpaint_image)
    app.router.add_post('/offer', offer)

    if cert_file is not None and key_file is not None:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(cert_file, key_file)
    else:
        ssl_context = None

    web.run_app(app, access_log=None, port=port, ssl_context=ssl_context, host=host)
