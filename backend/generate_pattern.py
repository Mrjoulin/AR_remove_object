from backend.makeGif import *
from backend.patchBasedTextureSynthesis import *
from PIL import Image

exampleMapPath = "pattern.jpg"

im = Image.open(exampleMapPath)
img = im.resize((200, 200), Image.ANTIALIAS)
img.save("pattern.jpg")

#PARAMS
outputPath = "out/1/"
patchSize = 30 #size of the patch (without the overlap)
overlapSize = 20 #the width of the overlap region
outputSize = [300, 300]

pbts = patchBasedTextureSynthesis(exampleMapPath, outputPath, outputSize, patchSize, overlapSize, in_windowStep=5,
                                  in_mirror_hor=True, in_mirror_vert=True, in_shapshots=True)
pbts.resolveAll()

gifOutputPath = "out/outGif.gif"

makeGif(outputPath, gifOutputPath, frame_every_X_steps=1, repeat_ending=5)