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
from server.frame_render import *

URL = "http://127.0.0.1:5000/"
# URL = "http://84.201.133.73:5000/"
ROOT = os.path.dirname(os.path.abspath(__file__))
pcs = set()

# Load object detection and inpaint model
RENDER = Render()


async def init(request):
    logging.info('Run init page from IP: %s' % request.remote)
    content = open(os.path.join(ROOT, "templates/test/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def init_js(request):
    content = open(os.path.join(ROOT, "templates/test/src/client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def test_inpaint(requset):
    test_json = json.loads(open('backend/test.json', 'r').read())
    imgs = make_api_request(url_server=URL, method_name='get_inpaint_image',
                            img=test_json['img'], objects=test_json['objects'])
    response = await imgs.json()
    logging.info('Return respose')
    return web.Response(text=str(response))


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
        video_transform: (check Input in Data Channel) - it's need to start
    :return:
        "boxes" - stream frames with visualized detected objects
        "inpaint" - stream frames with remove selected object

    Data Channel:
    Input:
        {
            "message_id": <message id>
            "name": <name_algorithm>, - name algorithm. Options: "", "edges", "boxes", "inpaint"
            "src": [<additional variables>] - for "inpaint" -- [<class id objects>] (For example: [1, 15]; ["all"])
                                              for others -- []
        }

    Output:
        {
            "message_id": <message id>,
            "data": [
                {
                    "class_id": <class id object>, - for example: 23 (int)
                    "position: {
                        "x_min": <position top left point of the rectangle>, - from the left edge (example: 0,1325..)
                        "y_min": <position top left point of the rectangle>, - from the top edge (example: 0,3271..)
                        "x_max": <position bottom right point of the rectangle>, - from the left edge (example: 0,562..)
                        "y_max": <position bottom right point of the rectangle> - from the top edge (example: 0,8932..)
                    }
                },
                ...

            ] - detected objects: for "inpaint" - removed objects
                                  for "boxes" - all detected objects
                                  for others - []
        }
    '''

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    logging.info(pc_id + " " + "Created for %s" % request.remote)

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

        local_video = (VideoTransformTrack(track, transform=params["video_transform"], render=RENDER))
        pc.addTrack(local_video)

        @pc.on("datachannel")
        def on_datachannel(channel):
            logging.info('Create Data Channel')
            algorithms_with_fps = {
                '': 0,
                'edges': 0,
                'cartoon': 0,
                'inpaint': 0,
                'boxes': 0
            }  # { "<name_algorithm>": <last_fps> }

            @channel.on("message")
            def on_message(message):

                message = json.loads(message)

                if isinstance(message, dict) and 'name' in message.keys() and \
                        message['name'] in algorithms_with_fps:
                    video_transform = message['name']

                    if video_transform != local_video.transform:
                        logging.info('Set new algorithm: %s' % (video_transform if video_transform else '" "'))
                        local_video.transform = video_transform

                    if video_transform == 'inpaint' and 'src' in message.keys() and \
                            isinstance(message['src'], list):
                        local_video.objects_to_remove = message['src']
                    else:
                        local_video.objects_to_remove = ["all"]

                    fps = max(int(1 / get_average_time_render(video_transform) - 0.5), 1)
                    if fps != algorithms_with_fps[video_transform]:
                        logging.info('New FPS in %s: %s' % (video_transform if video_transform else '" "', fps))
                        algorithms_with_fps[video_transform] = fps

                    # Send response in Data Channel 
                    response = {
                        "message_id": message["message_id"],
                        "fps": fps,
                        "data": local_video.objects
                    }

                    channel.send(str(response))  # json.dumps(local_video[0].objects)

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
    logging.info('Average time of one frame: %s sec' % get_average_time_render('all'))
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def run_app(port=5000, host=None, cert_file=None, key_file=None):
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', init)
    app.router.add_get('/src/client.js', init_js)
    app.router.add_get('/test_inpaint', test_inpaint)
    app.router.add_post('/get_inpaint_image', get_inpaint_image)
    app.router.add_post('/offer', offer)

    if cert_file is not None and key_file is not None:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(cert_file, key_file)
    else:
        ssl_context = None

    web.run_app(app, access_log=None, port=port, ssl_context=ssl_context, host=host)
