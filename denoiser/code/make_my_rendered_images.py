import data
from keras.preprocessing.image import array_to_img, img_to_array, load_img


with open("../data/full/report/reference_colour_0.txt") as f:
    ref_img = data.parseFileRGB(f) #** 0.9
    ref_img = array_to_img(ref_img)
    ref_img.save("../data/full/report/pathtracer65536.png")

