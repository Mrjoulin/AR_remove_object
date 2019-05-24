import logging
from AR_remover.objectdetection import find_object_in_image, camera
import cv2

logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)


def video_render(start_video_path, type_video, number_first_img=0, number_last_img=1):
    logging.info('Start render video - %s' % start_video_path)
    for number_img in range(number_first_img, number_last_img + 1):
        path = start_video_path + '000' + str(number_img) + '.' + type_video
        logging.info('Render - %s' % path)
        cap = cv2.VideoCapture(path)
        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        find_object_in_image(cap=cap, video_size=(vid_width, vid_height), render_video=True, number_video=number_img)


def only_camera_connection(video_path):
    # Web camera
    camera(video_path)


if __name__ == '__main__':
    # Ran an online rendering
    video_size = (800, 600)
    cap = cv2.VideoCapture(0)
    find_object_in_image(cap=cap, video_size=video_size)
