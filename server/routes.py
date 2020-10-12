import os
import cv2
import ssl
import json
import uuid
import time
import base64
import random
import asyncio
import logging
import requests
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

# local modules
from backend import source
from server.frame_render import *

ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_FPS = 30
pcs = set()

# Load object detection and inpaint model
RENDER = Render()
# Renders generated for image
# {
#   (<image_width>, <image_height>): Render()
# }
IMAGES_RENDERS = {}


async def init(request):
    logging.info('Run init page from IP: %s' % request.remote)
    content = open(os.path.join(ROOT, "templates/test/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def init_js(request):
    content = open(os.path.join(ROOT, "templates/test/src/client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


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

                    fps = min(max(int(1 / get_average_time_render(video_transform) - 0.5), 1), MAX_FPS)
                    if fps != algorithms_with_fps[video_transform]:
                        logging.info('New FPS in %s: %s' % (video_transform if video_transform else '""', fps))
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


def preload_image(image_path: str):
    image_np = cv2.imread(image_path)
    original_image_shape = image_shape = tuple(image_np.shape[1::-1])  # (width, height)
    aspect_ratio = image_shape[0] / image_shape[1]  # width / height
    find = False

    for shape in IMAGES_RENDERS:
        if shape[0] / shape[1] == aspect_ratio:
            logging.info("Find similar Render with shape %s. Original shape %s" % (str(shape), str(image_shape)))
            if shape[0] < image_shape[0] and shape[1] < image_shape[1]:
                IMAGES_RENDERS.pop(shape)
                IMAGES_RENDERS[image_shape] = Render(image_size=image_shape, run_test=False)
            elif shape[0] > image_shape[0] and shape[1] > image_shape[1]:
                image_np = cv2.resize(image_np, shape)
                image_shape = shape

            find = True
            break

    if not find:
        IMAGES_RENDERS[image_shape] = Render(image_size=image_shape, run_test=False)

    return original_image_shape, image_shape, image_np


def post_processing_info(info: dict, original_image_shape: tuple):

    if tuple(info["image"].shape[1::-1]) != original_image_shape:
        # Resize image to original size
        info["image"] = cv2.resize(info["image"], original_image_shape)

    success, encoded_image = cv2.imencode('.png', info["image"])
    info["image"] = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return info


async def predict_image(request):
    params = await request.json()

    logging.info("Get new post with keys: %s" % list(params))

    if "image" in params and isinstance(params["image"], str) and \
            "chat" in params and isinstance(params["chat"], int):
        image_base64 = params["image"]
        chat_id = params["chat"]
    else:
        return web.Response(
            text=json.dumps(
                {
                    "error": "Params is not valid"
                }
            ),
            status=500
        )

    image_id = random.randint(10**3, 10**8)
    image_path = "images/%s_%s.png" % (chat_id, image_id)
    with open(image_path, "wb") as image_file:
        image_file.write(
            base64.b64decode(image_base64.encode('utf-8'))
        )

    logging.info("Predict new image %s" % image_path)

    original_image_shape, render_image_shape, image_np = preload_image(image_path)

    info = IMAGES_RENDERS[render_image_shape].run(image=image_np, transform="boxes")

    logging.info("Find %s objects on image" % len(info["objects"]))

    info = post_processing_info(info, original_image_shape)
    info["image_id"] = image_id

    return web.Response(
        content_type="application/json",
        text=json.dumps(info)
    )


async def process_image(request):
    params = await request.json()

    logging.info("Get new post with params: %s" % params)

    if "image_id" in params and isinstance(params["image_id"], int) and \
            "chat" in params and isinstance(params["chat"], int) and \
            "objects_to_remove" in params and isinstance(params["objects_to_remove"], list):
        image_id = params["image_id"]
        chat_id = params["chat"]
        objects_to_remove = params["objects_to_remove"]
    else:
        return web.Response(
            text=json.dumps(
                {
                    "error": "Params is not valid"
                }
            ),
            status=400
        )

    image_path = "images/%s_%s.png" % (chat_id, image_id)

    if not os.path.exists(image_path):
        return web.Response(
            text=json.dumps(
                {
                    "error": "Saved image with given chat ID and image ID not found"
                }
            ),
            status=400
        )

    logging.info("Remove %s objects from image %s" % (len(objects_to_remove), image_path))

    original_image_shape, render_image_shape, image_np = preload_image(image_path)

    info = IMAGES_RENDERS[render_image_shape].inpaint_model.inpaint_image(img=image_np, objects=objects_to_remove)

    info = post_processing_info(info, original_image_shape)
    # Remove objects from return json
    info.pop("objects")

    return web.Response(
        content_type="application/json",
        text=json.dumps(info)
    )


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
    app.router.add_post('/offer', offer)
    app.router.add_post("/predict_image", predict_image)
    app.router.add_post("/process_image", process_image)

    if cert_file is not None and key_file is not None:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(cert_file, key_file)
    else:
        ssl_context = None

    web.run_app(app, access_log=None, port=port, ssl_context=ssl_context, host=host)
