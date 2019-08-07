import os
import cv2
import logging
import argparse
import absl.logging
from time import time

# Local modules
from AR_remover.objectdetection import *


logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)


def render_video_directory(render_directory, render_masking=False, render_inpaint=False, use_server=False, tf2=False):
    logging.info('Start render videos in  %s' % render_directory)
    render_start_time = time()
    try:
        videos = os.listdir(render_directory)
    except Exception as e:
        raise FileNotFoundError(e)

    videos.sort()
    if not render_masking and not render_inpaint:
        logging.warning('You have not selected no one render. If you want to continue, press ENTER, else press any key')
        success = input()
        if success:
            return None

    if render_directory[-1] != '/':
        render_directory = render_directory + '/'

    for number_video in range(len(videos)):
        path = render_directory + videos[number_video]
        logging.info('Render file %s' % videos[number_video])
        try:
            cap = cv2.VideoCapture(path)
        except:
            logging.error('Error for rendering file %s' % videos[number_video])
            continue

        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        tensorflow_render(cap=cap, video_size=(int(video_width), int(video_height)), use_server=use_server,
                          render_video_by_masking=render_masking, render_video_by_inpainting=render_inpaint,
                          number_video=number_video + 1, tf2=tf2)
    logging.info(f'---- Rendering {str(len(videos))} videos for {str(time() - render_start_time)} seconds ----')


def video_render(video_path, render_masking=False, render_inpaint=False, use_server=False, tf2=False):
    logging.info('Start render video - %s' % video_path)
    render_start_time = time()

    if not render_masking and not render_inpaint:
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
    tensorflow_render(cap=cap, video_size=(int(video_width), int(video_height)), use_server=use_server,
                      render_video_by_masking=render_masking, render_video_by_inpainting=render_inpaint, tf2=tf2)
    logging.info('---- Rendering video for %s seconds ----' % (time() - render_start_time))


def image_render(image_path, render_masking=False, render_inpaint=False, use_server=False, tf2=False):
    logging.info('Start rendering image in %s' % image_path)
    render_start_time = time()

    if not render_masking and not render_inpaint:
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
    tensorflow_render(cap=cap, video_size=video_size, use_server=use_server,
                      render_image_by_masking=render_masking, render_image_by_inpainting=render_inpaint, tf2=tf2)
    logging.info('---- Rendering image for %s seconds ----' % str(time() - render_start_time))


def online_render(use_server=False, tf2=False, opencv=False):
    # Ran an online rendering
    video_size = (800, 600)
    cap = cv2.VideoCapture(0)
    if opencv:
        opencv_render(cap=cap, video_size=video_size, use_server=use_server)
    else:
        tensorflow_render(cap=cap, video_size=video_size, use_server=use_server, tf2=tf2)


def only_camera_connection(video_path=0):
    logging.info('Only Web Camera Connection. Press q or Esc for destroy')
    # Web camera
    camera(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A True Thanos remover object in a video, image or online')
    parser.add_argument('--render_online', action='store_true', default=False, help='Online rendering')
    parser.add_argument('--render_directory', type=str, default='', help='Path a directory to render videos')
    parser.add_argument('--render_video', type=str, default='', help='Path a video to render')
    parser.add_argument('--render_image', type=str, default='', help='Path an image to render')
    parser.add_argument('--use_server', action='store_true', default=False, help='Use server for rendering image')
    parser.add_argument('--opencv', '--cv', action='store_true', default=False, help='Use a OpenCv to object '
                                                                                     'detection and masking')
    parser.add_argument('--tensorflow2', action='store_true', default=False, help='Use more accurate tensorflow')
    parser.add_argument('--inpaint', action='store_true', default=False, help='Use a inpaint rendering')
    parser.add_argument('--masking', action='store_true', default=False, help='Use a masking rendering')
    args = parser.parse_args()

    if args.render_directory:
        render_video_directory(render_directory=args.render_directory, use_server=args.use_server,
                               render_masking=args.masking, render_inpaint=args.inpaint, tf2=args.tensorflow2)
    elif args.render_video:
        video_render(video_path=args.render_video, use_server=args.use_server, tf2=args.tensorflow2,
                     render_masking=args.masking, render_inpaint=args.inpaint)
    elif args.render_image:
        image_render(image_path=args.render_image, use_server=args.use_server, tf2=args.tensorflow2,
                     render_masking=args.masking, render_inpaint=args.inpaint)
    elif args.render_online:
        online_render(use_server=args.use_server, tf2=args.tensorflow2, opencv=args.opencv)
    else:
        only_camera_connection()
