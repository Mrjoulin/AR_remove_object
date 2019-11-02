import os
import cv2
import time
import logging
import argparse
import absl.logging

# Local modules
from local.objectdetection import *


logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logging.basicConfig(
    format='[%(asctime)s: %(filename)s:%(lineno)s - %(funcName)10s()]%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


def render_video_directory(render_directory, inpaint=False, tf2=False):
    logging.info('Start render videos in  %s' % render_directory)
    render_start_time = time.time()
    try:
        videos = os.listdir(render_directory)
    except Exception as e:
        raise FileNotFoundError(e)

    videos.sort()
    if not inpaint:
        logging.warning('You have not selected no one render. If you want to continue, press ENTER, else press any key')
        success = input()
        if success:
            return None

    for number_video in range(len(videos)):
        path = os.path.join(render_directory, videos[number_video])
        logging.info('Render file %s' % videos[number_video])
        try:
            cap = cv2.VideoCapture(path)
        except:
            logging.error('Error for rendering file %s' % videos[number_video])
            continue

        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_size = (int(video_width), int(video_height))
        tensorflow_render(cap=cap, video_size=video_size, render_video=inpaint, number_video=number_video + 1, tf2=tf2)
    logging.info('---- Rendering %s videos for %s seconds ----' % (len(videos), (time.time() - render_start_time)))


def video_render(video_path, inpaint=False, tf2=False):
    logging.info('Start render video - %s' % video_path)
    render_start_time = time.time()

    if not inpaint:
        logging.warning('You have not selected no one render. If you want to continue, press ENTER, else press any key')
        success = input()
        if success:
            return None

    try:
        cap = cv2.VideoCapture(video_path)
    except:
        raise FileNotFoundError('Error for rendering file')

    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_size = (int(video_width), int(video_height))
    tensorflow_render(cap=cap, video_size=video_size, render_video=inpaint, tf2=tf2)
    logging.info('---- Rendering video for %s seconds ----' % (time.time() - render_start_time))


def image_render(image_path, inpaint=False, tf2=False):
    logging.info('Start rendering image in %s' % image_path)
    render_start_time = time.time()

    if not inpaint:
        logging.warning('You have not selected no one render. If you want to continue, press ENTER, else press any key')
        success = input()
        if success:
            return None

    if os.path.exists(image_path):
        try:
            cap = cv2.VideoCapture(image_path)
        except:
            raise FileNotFoundError('Error for rendering file')
    else:
        raise FileExistsError('No exists image file in %s' % image_path)

    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_size = (int(video_width), int(video_height))
    tensorflow_render(cap=cap, video_size=video_size, render_image=inpaint, tf2=tf2)
    logging.info('---- Rendering image for %s seconds ----' % str(time.time() - render_start_time))


def online_render(tf=False):
    # Ran an online rendering
    logging.info('Ran online rendering')
    video_size = (640, 480)
    cap = cv2.VideoCapture(0)
    if tf:
        tensorflow_render(cap=cap, video_size=video_size)
    else:
        tensorflow_with_trt_render(cap=cap, video_size=video_size)


def only_camera_connection(video_path=0):
    logging.info('Only Web Camera Connection. Press q or Esc for destroy')
    # Web camera
    camera(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A Blurred can remove object in a video, image or online')
    parser.add_argument('--online', action='store_true', default=False, help='Online rendering')
    parser.add_argument('--directory', type=str, default='', help='Path a directory to render videos')
    parser.add_argument('--video', type=str, default='', help='Path a video to render')
    parser.add_argument('--image', type=str, default='', help='Path an image to render')
    # parser.add_argument('--use_server', action='store_true', default=False, help='Use server for rendering image')
    # parser.add_argument('--opencv', '--cv', action='store_true', default=False, help='Use a OpenCv to object '
    #                                                                                  'detection and masking')
    parser.add_argument('--tensorflow', '-tf', action='store_true', default=False, help='Use more accurate tensorflow')
    parser.add_argument('--inpaint', action='store_true', default=False, help='Use a inpaint rendering')
    args = parser.parse_args()

    if args.directory:
        render_video_directory(render_directory=args.directory, inpaint=args.inpaint, tf2=args.tensorflow2)
    elif args.video:
        video_render(video_path=args.video, tf2=args.tensorflow2, inpaint=args.inpaint)
    elif args.image:
        image_render(image_path=args.image, tf2=args.tensorflow2, inpaint=args.inpaint)
    elif args.online:
        online_render(tf=args.tensorflow)
    else:
        only_camera_connection()
