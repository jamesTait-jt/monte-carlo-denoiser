import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np

import data

def patchify(img, patch_width, patch_height, img_width, img_height):
    img = img_to_array(img)
    num_patches = (img_width / patch_width) * (img_height / patch_height) 
    patches = np.zeros((int(num_patches), patch_height, patch_width, 3))
    ctr = 0
    for x in range(0, img_height - patch_height, patch_height):
        for y in range(0, img_width - patch_width, patch_width):
            patch = np.zeros((patch_height, patch_width, 3))
            for x1 in range(patch_height):
                for y1 in range(patch_height):
                    patch[x1][y1] = img[x + x1][y + y1]
            patches[ctr] = array_to_img(patch)
            ctr += 1
    return patches


PATCH_WIDTH = 64
PATCH_HEIGHT = 64
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# Load in trained model
model = tf.keras.models.load_model("models/model.h5")

# Load in the noisy image and feature buffers
noisy_colour = load_img("data/full/noisy_colour.png")
noisy_colour_gradx = load_img("data/full/noisy_colour_gradx.png") 
noisy_colour_grady = load_img("data/full/noisy_colour_grady.png") 
noisy_colour_var = array_to_img(data.convert_channels_3_to_1(img_to_array(load_img("data/full/noisy_colour_vars.png"))))
noisy_sn = load_img("data/full/noisy_surface_normal.png")
noisy_sn_gradx = load_img("data/full/noisy_surface_normal_gradx.png")
noisy_sn_grady = load_img("data/full/noisy_surface_normal_grady.png")

colour_patches = patchify(noisy_colour, PATCH_WIDTH, PATCH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
colour_gradx_patches = patchify(noisy_colour_gradx, PATCH_WIDTH, PATCH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
colour_grady_patches = patchify(noisy_colour_grady, PATCH_WIDTH, PATCH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
colour_var_patches = patchify(noisy_colour_var, PATCH_WIDTH, PATCH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
sn_patches = patchify(noisy_sn, PATCH_WIDTH, PATCH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
sn_gradx_patches = patchify(noisy_sn_gradx, PATCH_WIDTH, PATCH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
sn_grady_patches = patchify(noisy_sn_grady, PATCH_WIDTH, PATCH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)

model_input = np.concatenate(
        (
            colour_patches,
            colour_gradx_patches,
            colour_grady_patches,
            colour_var_patches,
            sn_patches,
            sn_gradx_patches,
            sn_grady_patches,
        ), 3)

pred = data.convert_channels_7_to_3(model.predict(model_input))


