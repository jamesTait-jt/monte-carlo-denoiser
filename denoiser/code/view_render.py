import sys
import exr

from keras.preprocessing.image import array_to_img, img_to_array

path = sys.argv[1]

exr_file = exr.open(path)

diffuse = exr.getBuffer(exr_file, "diffuse")
diffuse = diffuse.clip(0, 1)

img = array_to_img(diffuse)
img.show()
