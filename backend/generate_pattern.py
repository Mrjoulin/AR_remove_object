from backend.patchBasedTextureSynthesis import *
from PIL import Image


def get_generative_background(path_to_fragment, width_object, height_object, number_of_image):
    im = Image.open(path_to_fragment)
    img = im.resize((150, 150), Image.ANTIALIAS)
    img.save("backend/pattern.jpg")

    #PARAMS
    outputPath = "backend/out/1/"
    patchSize = 50 #size of the patch (without the overlap)
    overlapSize = 10 #the width of the overlap region
    outputSize = [300, 300]

    pbts = patchBasedTextureSynthesis(exampleMapPath='backend/pattern.jpg', in_outputPath=outputPath,
                                      in_outputSize=outputSize, in_patchSize=patchSize, in_overlapSize=overlapSize,
                                      number_of_image=number_of_image,
                                      in_windowStep=5, in_mirror_hor=False, in_mirror_vert=False, in_shapshots=False)
    pbts.resolveAll()

