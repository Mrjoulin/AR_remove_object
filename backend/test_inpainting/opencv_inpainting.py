from __future__ import print_function

from time import time
import numpy as np
import cv2 as cv
from PIL import Image

from backend.track_object.common import Sketcher

'''
Inpainting sample.
Inpainting repairs damage to images by floodfilling
the damage with surrounding image areas.
Usage:
  inpaint.py [<image>]
Keys:
  SPACE - inpaint
  r     - reset the inpainting mask
  ESC   - exit
'''


def main():
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'fruits.jpg'

    img = cv.imread(fn)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    sketch = Sketcher('img', [img_mark, mark], lambda: ((255, 255, 255), 255))

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            start_time = time()
            res = cv.inpaint(img_mark, mark, 3, cv.INPAINT_TELEA)
            print(start_time-time(), 'seconds')
            cv.imshow('inpaint big', res)
            start_time = time()
            res = cv.inpaint(img_mark, mark, 1, cv.INPAINT_TELEA)
            print(start_time - time(), 'seconds')
            cv.imshow('inpaint small', res)
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            sketch.show()

    print('Done')


if __name__ == '__main__':
    print('doc %s' % __doc__)
    main()
    cv.destroyAllWindows()
