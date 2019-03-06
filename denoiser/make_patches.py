import random
import numpy as np
from scipy import ndimage

from keras.preprocessing.image import array_to_img, img_to_array, load_img

import config

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



for i in range(len(config.IMAGE_PATHS_TO_DIFFERENTIATE)):

    img = load_img(config.IMAGE_PATHS_TO_DIFFERENTIATE[i])

    gradx = ndimage.sobel(img, axis=0, mode='constant')
    grady = ndimage.sobel(img, axis=1, mode='constant')

    gradx = array_to_img(gradx)
    grady = array_to_img(grady)

    gradx.save(config.DIFFERENTIATED_SAVE_DIRS[2 * i])
    grady.save(config.DIFFERENTIATED_SAVE_DIRS[2 * i + 1])

darts = generate_darts(
    config.NUM_DARTS,
    config.IMAGE_WIDTH,
    config.IMAGE_HEIGHT,
    config.PATCH_WIDTH,
    config.PATCH_HEIGHT
)

for i in range(len(config.FULL_IMAGE_PATHS)):
    img_array = img_to_array(load_img(config.FULL_IMAGE_PATHS[i]))

    # Each dart throw is the top left corner of the patch
    ctr = 0
    for dart in darts:
        ctr += 1
        new_patch = np.zeros((config.PATCH_WIDTH, config.PATCH_HEIGHT, 3))
        for x in range(0, config.PATCH_HEIGHT):
            for y in range(0, config.PATCH_WIDTH):
                new_patch[x][y] = img_array[dart[0] + x][dart[1] + y]
        new_patch = array_to_img(new_patch)
        new_patch.save(config.PATCH_SAVE_DIRS[i] + str(ctr) + ".png")

