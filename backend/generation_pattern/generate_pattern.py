from backend.generation_pattern.patchBasedTextureSynthesis import *
from PIL import Image
import logging


def get_generative_background(path_to_fragment, width_object, height_object, number_of_image):
    im = Image.open(path_to_fragment)
    bg_side = (int(im.width)) if im.width > im.height else (int(im.height))
    min_size = 20
    if bg_side > min_size:
        img = im.resize((bg_side, bg_side), Image.ANTIALIAS)
    else:
        img = im.resize((min_size, min_size), Image.ANTIALIAS)
    pattern_path = "backend/generation_pattern/pattern." + ('jpg' if img.mode == 'RGB' else 'png')
    logging.info('Image mode: %s' % img.mode)
    img.save(pattern_path)

    #PARAMS
    outputPath = "backend/out/1/"
    # size of the patch (without the overlap)
    patchSize = int((bg_side if bg_side < min_size else min_size) * 0.4)
    # the width of the overlap region
    overlapSize = patchSize // 4 if patchSize // 4 > 0 else 1

    outputSize = [int(width_object if width_object < height_object else height_object),
                  int(width_object if width_object < height_object else height_object)]

    logging.info('Generate pattern info: background squere size - {bgSize}, outputPath - {path}, patchSize - '
                 '{patchSize}, overlapSize {overlap}, outputSize - {output}'
                 .format(bgSize=bg_side, path=outputPath, patchSize=patchSize, overlap=overlapSize, output=outputSize))

    pbts = patchBasedTextureSynthesis(exampleMapPath=pattern_path, in_outputPath=outputPath,
                                      in_outputSize=outputSize, in_patchSize=patchSize, in_overlapSize=overlapSize,
                                      number_of_image=number_of_image,
                                      in_windowStep=5, in_mirror_hor=False, in_mirror_vert=False, in_shapshots=False)
    pbts.resolveAll()


if __name__ == '__main__':
    get_generative_background('backend/generation_pattern/test_img.jpg', 100, 100, 'stone_1')
    get_generative_background('backend/generation_pattern/test_img_2.jpg', 100, 100, 'stone_2')

