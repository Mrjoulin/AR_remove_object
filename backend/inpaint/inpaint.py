import os
import cv2
import time
import json
import logging
import argparse
import subprocess
import numpy as np
import neuralgym as ng
import tensorflow as tf
from argparse import Namespace

# Local imports
from backend.inpaint.inpaint_model import InpaintCAModel
from backend.inpaint.net.network import GMCNNModel

ROOT = os.path.dirname(os.path.abspath(__file__))
INPAINT_MODEL_DIR = os.path.join(ROOT, 'models/release_places2_256/')
NEWINPAINT_MODEL_DIR = os.path.join(ROOT, 'models/paris-streetview_256x256_rect')


class Inpainting:
    def __init__(self, session=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # ng.get_gpus(1)
        self.model = InpaintCAModel()
        if session is None:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.session = tf.Session(config=sess_config)
        else:
            self.session = session

    def load_model(self):
        load_model_time = time.time()
        logging.info('Using inpaint maodel: %s' % INPAINT_MODEL_DIR)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(INPAINT_MODEL_DIR, from_name)
            assign_ops.append(tf.assign(var, var_value))
        self.session.run(assign_ops)
        logging.info('Model loaded for %s sec' % (time.time() - load_model_time))

        return None

    def get_output(self, input_image):
        preload_time = time.time()
        output = self.model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        logging.info('Preload time: %s sec' % (time.time() - preload_time))

        return output


class NewInpainting:
    def __init__(self, session=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
            "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
            stdout=subprocess.PIPE).stdout.readlines()]))
        self.model = GMCNNModel()
        if session is None:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = False
            self.session = tf.Session(config=sess_config)
        else:
            self.session = session

    def load_model(self):
        start_time = time.time()
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(NEWINPAINT_MODEL_DIR, x.name)),
                              vars_list))
        self.session.run(assign_ops)
        logging.info('Model loaded for %s sec' % (time.time() - start_time))

    def get_output(self, image_tf, mask_tf, reuse):
        preload_time = time.time()
        logging.info('Using model: %s' % NEWINPAINT_MODEL_DIR)
        config = Namespace(d_cnum=64, dataset='places2', random_mask=False, seed=1, test_num=-1, g_cnum=32,
                           img_shapes=[480, 640, 3], load_model_dir=NEWINPAINT_MODEL_DIR, mask_shapes=[480, 640],
                           mask_type='rect', mode='save', model='gmcnn', model_prefix='snap')

        output = self.model.evaluate(image_tf, mask_tf, config=config, reuse=reuse)
        output = (output + 1) * 127.5
        output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
        output = tf.cast(output, tf.uint8)
        logging.info('Preload time: %s sec' % (time.time() - preload_time))
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default='', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default='output.png', type=str,
                        help='Where to write output.')
    args = parser.parse_args()

    #model = Inpainting()
    #response = model.get_output(cv2.imread(args.image), cv2.imread(args.mask), reuse=False)

    # Load pretrained model
    #model.load_model()

    #frame_time = time.time()
    #result = model.session.run(response)
    #cv2.imwrite(args.output, result[0][:, :, ::-1])

    inpaint_session = NewInpainting()
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask, 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, image.shape[0], image.shape[1], 3])
    input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, image.shape[0], image.shape[1], 1])
    output = inpaint_session.get_output(input_image_tf, input_mask_tf, reuse=False)
    inpaint_session.load_model()
    frame_time = time.time()
    image = image * (1 - mask / 255) + mask
    cv2.imwrite(args.output.replace('.png', '_input.png'), image.astype(np.uint8))
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask / 255, 0)
    result = inpaint_session.session.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
    cv2.imwrite(args.output, result[0][:, :, ::-1])
    print('---- Frame time %s sec ----' % (time.time() - frame_time))
    # response = model.get_output(cv2.imread("server/imgs/inpaint_480.png"), cv2.imread("server/imgs/mask_480.png"),
    #                            reuse=tf.AUTO_REUSE)
    # frame_time = time.time()
    # result = model.session.run(response)
    # cv2.imwrite(args.output, result[0][:, :, ::-1])
    # print('---- Frame time %s sec ----' % (time.time() - frame_time))
