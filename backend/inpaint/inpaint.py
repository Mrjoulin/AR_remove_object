import os
import cv2
import time
import shutil
import logging
import argparse
import subprocess
import numpy as np
import neuralgym as ng
import tensorflow as tf

# Local imports
from backend.inpaint.net.v2.inpaint_model import InpaintCAModel

ROOT = os.path.dirname(os.path.abspath(__file__))
INPAINT_MODEL_DIR = os.path.join(ROOT, 'models/release_places2_v2/')

MODELS = {
    'places2_v1': {
        'url': 'https://drive.google.com/uc?export=download&confirm=QjRw&id=1aakVS0CPML_Qg-PuXGE1Xaql96hNEKOU',
        'name': 'places2_512x680_freeform.zip'
    },
    'places2_v2': {
        'url': [
            'https://drive.google.com/uc?export=download&id=1dyPD2hx0JTmMuHYa32j-pu--MXqgLgMy',
            'https://drive.google.com/uc?export=download&confirm=bL_Y&id=1z9dbEAzr5lmlCewixevFMTVBmNuSNAgK',
            'https://drive.google.com/uc?export=download&confirm=bL_Y&id=1ExY4hlx0DjVElqJlki57la3Qxu40uhgd',
            'https://drive.google.com/uc?export=download&confirm=bL_Y&id=1C7kPxqrpNpQF7B2UAKd_GvhUMk0prRdV'
        ],
        'name': [
            'checkpoint',
            'snap-0.data-00000-of-00001',
            'snap-0.index',
            'snap-0.meta'
        ]
    }
}


def download_model(model_name, input_dir=None):
    model = MODELS[model_name]

    if type(model['url']) is list:
        subprocess.call(['mkdir', '-p', input_dir])
        for url, name in zip(model['url'], model['name']):
            subprocess.call(['wget', '-q', url, '-O', os.path.join(input_dir, name)])
    else:
        if not os.path.exists(input_dir):
            subprocess.call(['mkdir', '-p', input_dir])
        print('Downloading model from: %s' % model['url'])
        subprocess.call(['wget', '-q', model['url'], '-O', os.path.join(input_dir, model['name'])])
        subprocess.call(['tar', '-xzf', os.path.join(input_dir, model['name']), '-C', input_dir])
        subprocess.call(['rm', '-rf', os.path.join(input_dir, model['name'])])


class Inpainting:
    def __init__(self, session=None):
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
        logging.info('Using inpaint model: %s' % INPAINT_MODEL_DIR)
        if not os.path.exists(INPAINT_MODEL_DIR):
            download_model('places2_v2', INPAINT_MODEL_DIR)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(INPAINT_MODEL_DIR, from_name)
            assign_ops.append(tf.assign(var, var_value))
        self.session.run(assign_ops)
        logging.info('Model loaded for %.5f sec' % (time.time() - load_model_time))

        return None

    def get_output(self, input_image):
        preload_time = time.time()
        FLAG = {'guided': False, 'edge_threshold': 0.5}
        output = self.model.build_server_graph(FLAG, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        logging.info('Preload time: %.5f sec' % (time.time() - preload_time))

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

    inpaint_model = Inpainting()

    size = (640, 480)
    small_size = (320, 240)
    input_image_tf = tf.placeholder(
        dtype=tf.float32,
        shape=(1, small_size[1], small_size[0] * 2, 3)
    )
    _output = inpaint_model.get_output(input_image_tf)
    inpaint_model.load_model()
    # Test run algorithm detection and inpainting (to overclock model)
    # Load test image
    img = cv2.resize(cv2.imread(args.image), size)
    init_img = img.copy()
    img = cv2.resize(img, small_size)
    # Remove needed objects by inpaint algorithm
    frame_time = time.time()
    # Get mask objects
    mask = cv2.resize(cv2.imread(args.mask), small_size) // 255
    # Inpaint image
    img = img * (1 - mask) + 255 * mask
    img = np.expand_dims(img, 0)
    input_mask = np.expand_dims(255 * mask, 0)
    input_image = np.concatenate([img, input_mask], axis=2)
    result = inpaint_model.session.run(_output, feed_dict={input_image_tf: input_image})

    # Merge the result of program to initial image
    img = result[0][:, :, ::-1]
    initial_size = (init_img.shape[1], init_img.shape[0])
    big_mask = cv2.resize(mask, initial_size)
    inpaint_objects = cv2.resize(img, initial_size) * big_mask
    image_np = init_img * (1 - big_mask)
    img = image_np + inpaint_objects
    print('---- Frame time %.5f sec ----' % (time.time() - frame_time))
    cv2.imwrite(args.output, img)

    '''
    inpaint_session = Inpainting()
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask, 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, image.shape[0], image.shape[1], 3])
    input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, image.shape[0], image.shape[1], 1])
    output = inpaint_session.get_output(input_image_tf)
    inpaint_session.load_model()
    image = image * (1 - mask / 255) + mask
    cv2.imwrite(args.output.replace('.%s' % args.output.split('.')[-1], '_input.png'), image.astype(np.uint8))
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask / 255, 0)
    result = inpaint_session.session.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
    cv2.imwrite(args.output, result[0][:, :, ::-1])
    '''

    # response = model.get_output(cv2.imread("server/imgs/inpaint_480.png"), cv2.imread("server/imgs/mask_480.png"),
    #                            reuse=tf.AUTO_REUSE)
    # frame_time = time.time()
    # result = model.session.run(response)
    # cv2.imwrite(args.output, result[0][:, :, ::-1])
    # print('---- Frame time %s sec ----' % (time.time() - frame_time))
