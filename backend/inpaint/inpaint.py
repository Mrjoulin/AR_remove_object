import os
import cv2
import time
import logging
import argparse
import numpy as np
import neuralgym as ng
import tensorflow as tf

# Local imports
from backend.inpaint.inpaint_model import InpaintCAModel

ROOT = os.path.dirname(os.path.abspath(__file__))
INPAINT_MODEL_DIR = ROOT + 'models/release_places2_256/'


class Inpainting:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # ng.get_gpus(1)
        self.model = InpaintCAModel()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=sess_config)

    def load_model(self):
        load_model_time = time.time()
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

    def get_output(self, image, mask, reuse):
        preload_time = time.time()
        logging.info('Image shape:', image.shape)
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        _input_image = tf.constant(input_image, dtype=tf.float32)
        output = self.model.build_server_graph(_input_image, reuse=reuse)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
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

    model = Inpainting()
    response = model.get_output(cv2.imread(args.image), cv2.imread(args.mask), reuse=False)

    # Load pretrained model
    model.load_model()

    frame_time = time.time()
    result = model.session.run(response)
    cv2.imwrite(args.output, result[0][:, :, ::-1])
    print('---- Frame time %s sec ----' % (time.time() - frame_time))
