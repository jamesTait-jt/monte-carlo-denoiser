import random
import numpy as np

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

NUM_DARTS = 200
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
PATCH_WIDTH = 64
PATCH_HEIGHT = 64

IMAGE_PATHS = [
    "data/full/noisy.jpg",
    "data/full/reference.jpg"
]

SAVE_DIRS = [
    "data/patches/noisy/",
    "data/patches/reference/"
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
        new_patch.save(SAVE_DIRS[i] + str(ctr) + ".jpg")




