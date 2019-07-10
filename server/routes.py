import numpy as np
import requests
import logging
import base64
import cv2
import json
import os
import uuid
import time
import asyncio
from aiohttp import web
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

# local modules
from backend import source


URL = "http://127.0.0.1:5000/"
# URL = "http://94.103.94.220:5000/"
ROOT = os.path.dirname(os.path.abspath(__file__))
pcs = set()


async def init(request):
    logging.info('Run init page')
    content = open(os.path.join(ROOT, "templates/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


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


def get_image(request_json, masking=False, inpaint=False):
    if 'img' in request_json and isinstance(request_json['img'], str) and \
            'objects' in request_json and isinstance(request_json['objects'], list):
        img = request_json['img']
        objects = request_json['objects']
        if masking and 'class_objects' in request_json and isinstance(request_json['class_objects'], list) and \
            len(request_json['objects']) == len(request_json['class_objects']):
            class_objects = request_json['class_objects']
    else:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'message': 'Partial Content'}
            ),
        )

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
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'img': encoded_image.decode("utf-8")}
            ),
        )

    except Exception as e:
        logging.error(e)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {'message': 'Internal Server Error'}
            ),
        )


def connect_to_tensorflow_graph():
    PATH_TO_FROZEN_GRAPH = "./AR_remover/objectdetection/tensorflow-graph/mask_rcnn/frozen_inference_graph.pb"
    PATH_TO_LABELS = './AR_remover/objectdetection/tensorflow-graph/mask_rcnn/mscoco_label_map.pbtxt'

    render_time = time.time()
    net = cv2.dnn.readNetFromTensorflow(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    logging.info('Connecting Tensorflow model in %s sec' % (time.time() - render_time))
    return net


def object_detection(net, img, box=False, mask=False):
    if box or mask:
        render_time = time.time()
        blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)
        # Set the input to the network
        net.setInput(blob)

        logging.info('Start detection')
        # Run the forward pass to get output from the output layers
        if box and mask:
            boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
            logging.info('Render image in %s' % (time.time() - render_time))
            return boxes, masks
        elif box:
            boxes = net.forward(['detection_out_final'])
            logging.info('Render image in %s' % (time.time() - render_time))
            return boxes
        else:
            masks = net.forward(['detection_masks'])
            logging.info('Render image in %s' % (time.time() - render_time))
            return masks

    return None


class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.net = connect_to_tensorflow_graph()

    async def recv(self):
        start_time = time.time()
        frame = await self.track.recv()
        if self.transform == 'boxes' or self.transform == 'inpaint':
            img = frame.to_ndarray(format="bgr24")

            if self.transform == "inpaint":
                logging.info('Inpaint image')

                masks, boxes = object_detection(self.net, img, box=True, mask=True)

                new_img = source.get_image_inpaint(img, masks=masks, boxes=boxes)
            else:
                logging.info('Go to draw boxes')

                boxes = object_detection(self.net, img, box=True)

                new_img = source.postprocess(frame=img, boxes=boxes, draw=True)

            new_frame = VideoFrame.from_ndarray(new_img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            logging.info('Return frame in %s' % (time.time() - start_time))

            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            logging.info('Return frame in %s' % (start_time - time.time()))
            return new_frame
        else:
            return frame


async def offer(request):
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


def run_app(port=5000, host=None):
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', init)
    app.router.add_get('/test_masking', test_masking)
    app.router.add_get('/test_inpaint', test_inpaint)
    app.router.add_post('/get_masking_image', get_masking_image)
    app.router.add_post('/get_inpaint_image', get_inpaint_image)
    app.router.add_post('/offer', offer)
    web.run_app(app, access_log=None, port=port, ssl_context=None, host=host)
