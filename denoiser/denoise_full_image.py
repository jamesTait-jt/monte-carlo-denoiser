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
    for x in range(0, img_height - patch_height + 1, patch_height):
        for y in range(0, img_width - patch_width + 1, patch_width):
            patch = np.zeros((patch_height, patch_width, 3))
            for x1 in range(patch_height):
                for y1 in range(patch_height):
                    patch[x1][y1] = img[x + x1][y + y1]
            patches[ctr] = array_to_img(patch)
            ctr += 1
    return patches

def stitch(img, patch_width, patch_height, img_width, img_height):
    patches_per_row = int(img_width / patch_width)
    patches_per_col = int(img_height / patch_height)
    img2d = np.zeros((img_height, img_width, 3))
    for i in range(img.shape[0]):
        x = patch_height * (i % patches_per_row)
        y = patch_width * (i // patches_per_row) #integer division
        for x1 in range(img[i].shape[0]):
            for y1 in range(img[i].shape[1]):
                img2d[y + y1][x + x1][0] = img[i][y1][x1][0]
                img2d[y + y1][x + x1][1] = img[i][y1][x1][1]
                img2d[y + y1][x + x1][2] = img[i][y1][x1][2]
    return array_to_img(img2d)


PATCH_WIDTH = 64
PATCH_HEIGHT = 64
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# Load in trained model
model = tf.keras.models.load_model("models/model.h5")

# Load in the noisy image and feature buffers
noisy_colour = load_img("data/full/noisy_colour.png")
noisy_colour_gradx = load_img("data/full/noisy_colour_gradx.png") 
noisy_colour_grady = load_img("data/full/noisy_colour_grady.png") 
noisy_sn = load_img("data/full/noisy_sn.png")
noisy_sn_gradx = load_img("data/full/noisy_sn_gradx.png")
noisy_sn_grady = load_img("data/full/noisy_sn_grady.png")
noisy_albedo = load_img("data/full/noisy_albedo.png")
noisy_albedo_gradx = load_img("data/full/noisy_albedo_gradx.png")
noisy_albedo_grady = load_img("data/full/noisy_albedo_grady.png")
noisy_depth = load_img("data/full/noisy_depth.png")
noisy_depth_gradx = load_img("data/full/noisy_depth_gradx.png")
noisy_depth_grady = load_img("data/full/noisy_depth_grady.png")
noisy_colour_var = load_img("data/full/noisy_colour_vars.png")
noisy_sn_var = load_img("data/full/noisy_sn_vars.png")
noisy_albedo_var = load_img("data/full/noisy_albedo_vars.png")
noisy_depth_var = load_img("data/full/noisy_depth_vars.png")

colour_patches = patchify(
    noisy_colour, 
    PATCH_WIDTH, 
    PATCH_HEIGHT, 
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

colour_gradx_patches = patchify(
    noisy_colour_gradx, 
    PATCH_WIDTH, 
    PATCH_HEIGHT, 
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

colour_grady_patches = patchify(
    noisy_colour_grady, 
    PATCH_WIDTH, 
    PATCH_HEIGHT, 
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

sn_patches = patchify(
    noisy_sn, 
    PATCH_WIDTH, 
    PATCH_HEIGHT, 
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

sn_gradx_patches = patchify(
    noisy_sn_gradx, 
    PATCH_WIDTH, 
    PATCH_HEIGHT, 
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

sn_grady_patches = patchify(
    noisy_sn_grady, 
    PATCH_WIDTH,
    PATCH_HEIGHT,         
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

albedo_patches = patchify(
    noisy_albedo, 
    PATCH_WIDTH, 
    PATCH_HEIGHT, 
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

albedo_gradx_patches = patchify(
    noisy_albedo_gradx, 
    PATCH_WIDTH, 
    PATCH_HEIGHT, 
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

albedo_grady_patches = patchify(
    noisy_albedo_grady, 
    PATCH_WIDTH,
    PATCH_HEIGHT,         
    IMAGE_WIDTH, 
    IMAGE_HEIGHT
)

depth_patches = data.convert_channels_3_to_1(
    patchify(
        noisy_depth, 
        PATCH_WIDTH, 
        PATCH_HEIGHT, 
        IMAGE_WIDTH, 
        IMAGE_HEIGHT
    )
)

depth_gradx_patches = data.convert_channels_3_to_1(
    patchify(
        noisy_depth_gradx, 
        PATCH_WIDTH, 
        PATCH_HEIGHT, 
        IMAGE_WIDTH, 
        IMAGE_HEIGHT
    )
)

depth_grady_patches = data.convert_channels_3_to_1(
    patchify(
        noisy_depth_grady, 
        PATCH_WIDTH,
        PATCH_HEIGHT,         
        IMAGE_WIDTH, 
        IMAGE_HEIGHT
    )
)

colour_var_patches = data.convert_channels_3_to_1(
    patchify(
        noisy_colour_var, 
        PATCH_WIDTH, 
        PATCH_HEIGHT, 
        IMAGE_WIDTH, 
        IMAGE_HEIGHT
    )
)

sn_var_patches = data.convert_channels_3_to_1(
    patchify(
        noisy_sn_var, 
        PATCH_WIDTH, 
        PATCH_HEIGHT, 
        IMAGE_WIDTH, 
        IMAGE_HEIGHT
    )
)

albedo_var_patches = data.convert_channels_3_to_1(
    patchify(
        noisy_albedo_var, 
        PATCH_WIDTH, 
        PATCH_HEIGHT, 
        IMAGE_WIDTH, 
        IMAGE_HEIGHT
    )
)

depth_var_patches = data.convert_channels_3_to_1(
    patchify(
        noisy_depth_var, 
        PATCH_WIDTH, 
        PATCH_HEIGHT, 
        IMAGE_WIDTH, 
        IMAGE_HEIGHT
    )
)

model_input = np.concatenate(
    (
        colour_patches,
        colour_gradx_patches,
        colour_grady_patches,
        #sn_patches,
        sn_gradx_patches,
        sn_grady_patches,
        #albedo_patches,
        albedo_gradx_patches,
        albedo_grady_patches,
        #depth_patches,
        depth_gradx_patches,
        depth_grady_patches,
        colour_var_patches,
        sn_var_patches,
        albedo_var_patches,
        depth_var_patches
    ), 3)


pred = data.convert_channels_7_to_3(model.predict(model_input))

stitched = stitch(pred, PATCH_WIDTH, PATCH_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
stitched.save("data/output/denoised.png")
