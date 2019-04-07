from backend.makeGif import *
from backend.patchBasedTextureSynthesis import *
from PIL import Image

exampleMapPath = "backend/pattern.jpg"

im = Image.open(exampleMapPath)
img = im.resize((150, 150), Image.ANTIALIAS)
img.save("backend/pattern.jpg")

#PARAMS
outputPath = "backend/out/1/"
patchSize = 30 #size of the patch (without the overlap)
overlapSize = 10 #the width of the overlap region
outputSize = [300, 300]

pbts = patchBasedTextureSynthesis(exampleMapPath, outputPath, outputSize, patchSize, overlapSize, in_windowStep=5,
                                  in_mirror_hor=True, in_mirror_vert=True, in_shapshots=False)
pbts.resolveAll()

gifOutputPath = "backend/out/outGif.gif"
