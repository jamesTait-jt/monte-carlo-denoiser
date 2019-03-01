import random
import numpy as np
from scipy import ndimage

from keras.preprocessing.image import array_to_img, img_to_array, load_img

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

### We need to calculate the image gradients of the colour data ###

IMAGE_PATHS_TO_DIFFERENTIATE = [
    "data/full/reference_colour.png",
    "data/full/noisy_colour.png",
    "data/full/reference_surface_normal.png",
    "data/full/noisy_surface_normal.png"
]

DIFFERENTIATED_SAVE_DIRS = [
    "data/full/reference_colour_gradx.png",
    "data/full/reference_colour_grady.png",
    "data/full/noisy_colour_gradx.png",
    "data/full/noisy_colour_grady.png",
    "data/full/reference_surface_normal_gradx.png",
    "data/full/reference_surface_normal_grady.png",
    "data/full/noisy_surface_normal_gradx.png",
    "data/full/noisy_surface_normal_grady.png"
]

for i in range(len(IMAGE_PATHS_TO_DIFFERENTIATE)):

    img = load_img(IMAGE_PATHS_TO_DIFFERENTIATE[i])

    gradx = ndimage.sobel(img, axis=0, mode='constant')
    grady = ndimage.sobel(img, axis=1, mode='constant')

    gradx = array_to_img(gradx)
    grady = array_to_img(grady)

    gradx.save(DIFFERENTIATED_SAVE_DIRS[2 * i])
    grady.save(DIFFERENTIATED_SAVE_DIRS[2 * i + 1])


NUM_DARTS = 600
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
PATCH_WIDTH = 64
PATCH_HEIGHT = 64

IMAGE_PATHS = [
    "data/full/reference_colour.png",
    "data/full/reference_colour_gradx.png",
    "data/full/reference_colour_grady.png",
    "data/full/noisy_colour.png",
    "data/full/noisy_colour_gradx.png",
    "data/full/noisy_colour_grady.png",
    "data/full/reference_colour_vars.png",
    "data/full/noisy_colour_vars.png",
    "data/full/reference_surface_normal.png",
    "data/full/reference_surface_normal_gradx.png",
    "data/full/reference_surface_normal_grady.png",
    "data/full/noisy_surface_normal.png",
    "data/full/noisy_surface_normal_gradx.png",
    "data/full/noisy_surface_normal_grady.png"
]

SAVE_DIRS = [
    "data/patches/reference_colour/",
    "data/patches/reference_colour_gradx/",
    "data/patches/reference_colour_grady/",
    "data/patches/noisy_colour/",
    "data/patches/noisy_colour_gradx/",
    "data/patches/noisy_colour_grady/",
    "data/patches/reference_colour_vars/",
    "data/patches/noisy_colour_vars/",
    "data/patches/reference_surface_normal/",
    "data/patches/reference_surface_normal_gradx/",
    "data/patches/reference_surface_normal_grady/",
    "data/patches/noisy_surface_normal/",
    "data/patches/noisy_surface_normal_gradx/",
    "data/patches/noisy_surface_normal_grady/"
]

darts = generate_darts(
    NUM_DARTS,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    PATCH_WIDTH,
    PATCH_HEIGHT
)

for i in range(len(IMAGE_PATHS)):
    img_array = img_to_array(load_img(IMAGE_PATHS[i]))

    # Each dart throw is the top left corner of the patch
    ctr = 0
    for dart in darts:
        ctr += 1
        new_patch = np.zeros((PATCH_WIDTH, PATCH_HEIGHT, 3))
        for x in range(0, PATCH_HEIGHT):
            for y in range(0, PATCH_WIDTH):
                new_patch[x][y] = img_array[dart[0] + x][dart[1] + y]
        new_patch = array_to_img(new_patch)
        new_patch.save(SAVE_DIRS[i] + str(ctr) + ".png")

