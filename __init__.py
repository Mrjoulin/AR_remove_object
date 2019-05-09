import logging
from AR_remover.objectdetection import find_object_in_image

logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)
 
if __name__ == '__main__':
    find_object_in_image()
