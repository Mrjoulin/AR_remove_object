import os
import cv2
import time
import shutil
import logging
import argparse
import absl.logging

# Local modules
from local.render import *

logger = logging.getLogger()

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

if os.path.exists('./neuralgym_logs'):
    shutil.rmtree('./neuralgym_logs')

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logging.basicConfig(
    format='[%(asctime)s: %(filename)s:%(lineno)s - %(funcName)10s()]%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


def render_video_directory(render_directory, inpaint=True, output='./', tf=False):
    logging.info('Start render videos in  %s' % render_directory)
    render_start_time = time.time()
    try:
        videos = os.listdir(render_directory)
    except Exception as e:
        raise FileNotFoundError(e)

    videos.sort()
    if not inpaint:
        logging.warning('You have not selected no one render. If you want to continue, ress ENTER, else press any key')
        success = input()
        if success:
            return None

    if not os.path.exists(output):
        subprocess.call(['mkdir', '-f', output])

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
        if tf:
            tensorflow_render(cap=cap, video_size=video_size, render_video=inpaint, number_video=number_video + 1)
        else:
            trt_render(cap=cap, video_path=path, output=output)
    logging.info('---- Rendering %s videos for %s seconds ----' % (len(videos), (time.time() - render_start_time)))


def video_render(video_path, inpaint=True, output='./', tf=False):
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

    if not os.path.exists(output):
        subprocess.call(['mkdir', '-f', output])

    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_size = (int(video_width), int(video_height))
    if tf:
        tensorflow_render(cap=cap, video_size=video_size, render_video=inpaint, tf2=tf)
    else:
        trt_render(cap=cap, video_path=video_path, output=output)
    logging.info('---- Rendering video for %s seconds ----' % (time.time() - render_start_time))


def image_render(image_path, inpaint=False, output='./', tf=False):
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

    if not os.path.exists(output):
        subprocess.call(['mkdir', '-f', output])

    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_size = (int(video_width), int(video_height))
    if tf:
        tensorflow_render(cap=cap, video_size=video_size, render_image=inpaint, tf2=tf)
    else:
        tensorflow_with_trt_render(cap=cap, video_size=video_size, image=True, output=output)
    logging.info('---- Rendering image for %s seconds ----' % str(time.time() - render_start_time))


def online_render(tf=False):
    # Ran an online rendering
    logging.info('Run online rendering')
    video_size = (640, 480)
    cap = cv2.VideoCapture(0)
    if tf:
        tensorflow_render(cap=cap, video_size=video_size)
    else:
        trt_render(cap=cap)


def only_camera_connection(video_path=0):
    logging.info('Only Web Camera Connection. Press q or Esc for destroy')
    # Web camera
    camera(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A Blurred can remove object in a video, image or online')
    parser.add_argument('--online', '-o', action='store_true', default=False, help='Online rendering')
    parser.add_argument('--directory', '-d', type=str, default='', help='Path a directory to render videos')
    parser.add_argument('--video', '-v', type=str, default='', help='Path a video to render')
    parser.add_argument('--image', '-im', type=str, default='', help='Path an image to render')
    parser.add_argument('--output', '-out', type=str, default='./', help='Output directory')
    # parser.add_argument('--use_server', action='store_true', default=False, help='Use server for rendering image')
    # parser.add_argument('--opencv', '--cv', action='store_true', default=False, help='Use a OpenCv to object '
    #                                                                                  'detection and masking')
    parser.add_argument('--tensorflow', '-tf', action='store_true', default=False, help='Use more accurate tensorflow')
    parser.add_argument('--inpaint', '-i', action='store_true', default=True, help='Use a inpaint rendering')
    args = parser.parse_args()

    if args.directory:
        render_video_directory(render_directory=args.directory, inpaint=args.inpaint, tf=args.tensorflow, output=args.output)
    elif args.video:
        video_render(video_path=args.video, tf=args.tensorflow, output=args.output, inpaint=args.inpaint)
    elif args.image:
        image_render(image_path=args.image, tf=args.tensorflow, output=args.output, inpaint=args.inpaint)
    elif args.online:
        online_render(tf=args.tensorflow)
    else:
        only_camera_connection()
