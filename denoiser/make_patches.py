import random
import numpy as np
from scipy import ndimage

from keras.preprocessing.image import array_to_img, img_to_array, load_img

import config
import new_data

def generate_darts(num_darts, image_width, image_height, patch_width, patch_height):
    x_lower = 0
    x_upper = image_height - patch_height

    y_lower = 0
    y_upper = image_width - patch_width

    darts = []
    for _ in range(num_darts):
        rand_x = random.randint(x_lower, x_upper)
        rand_y = random.randint(y_lower, y_upper)
        darts.append((rand_x, rand_y))

    return darts

darts = generate_darts(
    config.NUM_DARTS,
    config.IMAGE_WIDTH,
    config.IMAGE_HEIGHT,
    config.PATCH_WIDTH,
    config.PATCH_HEIGHT
)

patches = {}

i = 0
for key in new_data.full_images:
    img_array = new_data.full_images[key]

    if "depth" in key.split('_') or key.endswith("var"):
        channels = 1
    else:
        channels = 3

    # List of all patches for this key
    new_patches = []

    # Each dart throw is the top left corner of the patch
    ctr = 0
    for dart in darts:
        ctr += 1
        new_patch = np.zeros((config.PATCH_WIDTH, config.PATCH_HEIGHT, channels))
        for x in range(0, config.PATCH_HEIGHT):
            for y in range(0, config.PATCH_WIDTH):
                # fill in the patch pixel by pixel
                new_patch[x][y] = img_array[dart[0] + x][dart[1] + y]
        # Add the new patch to the array of patches for this key
        new_patches.append(np.array(new_patch))
        #new_patch = array_to_img(new_patch)
        #new_patch.save(config.PATCH_SAVE_DIRS[i] + str(ctr) + ".png")

    # Add all patches for this key to the dictionary
    patches[key] = np.array(new_patches)
    i += 1
