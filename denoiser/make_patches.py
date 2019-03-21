import random
import numpy as np
from scipy import ndimage
import random

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

# Gets a list of random floats in (1-limit, 1+limit) to apply to reference and
# noisy colour patches
def generateBrightnessFactors(limit):
    factors = [random.uniform(-limit, limit) for i in range(config.TRAIN_SCENES * config.NUM_DARTS)]
    return factors

def makePatches(seed):
    random.seed(seed)
    darts = generate_darts(
        config.NUM_DARTS,
        config.IMAGE_WIDTH,
        config.IMAGE_HEIGHT,
        config.PATCH_WIDTH,
        config.PATCH_HEIGHT
    )

    patches = {
        "train" : {
            "reference_colour" : [],
            "noisy_colour" : [],
            "noisy_colour_gradx" : [],
            "noisy_colour_grady" : [],
            "noisy_colour_var" : [],
            "noisy_sn_gradx" : [],
            "noisy_sn_grady" : [],
            "noisy_sn_var" : [],
            "noisy_albedo_gradx" : [],
            "noisy_albedo_grady" : [],
            "noisy_albedo_var" : [],
            "noisy_depth_gradx" : [],
            "noisy_depth_grady" : [],
            "noisy_depth_var" : []
        },
        "test" : {
            "reference_colour" : [],
            "noisy_colour" : [],
            "noisy_colour_gradx" : [],
            "noisy_colour_grady" : [],
            "noisy_colour_var" : [],
            "noisy_sn_gradx" : [],
            "noisy_sn_grady" : [],
            "noisy_sn_var" : [],
            "noisy_albedo_gradx" : [],
            "noisy_albedo_grady" : [],
            "noisy_albedo_var" : [],
            "noisy_depth_gradx" : [],
            "noisy_depth_grady" : [],
            "noisy_depth_var" : []
        },
    }

    full_images = new_data.loadAndPreProcessImages()
    brightness_factors = generateBrightnessFactors(0.3)

    print("Generating patches...")
    for test_or_train in full_images:
        augmentation = True
        if (test_or_train == "train"):
            train_dir = "train/"
            num_scenes = config.TRAIN_SCENES
        else:
            train_dir = "test/"
            num_scenes = config.TEST_SCENES
            augmentation = False

        full_img_num = 0 
        for key in full_images[test_or_train]:

            if key == "reference_colour":
                label = 1
            else:
                label = 0

            new_patches = []
            for i in range(num_scenes):
                img_array = full_images[test_or_train][key][i]

                if "depth" in key.split('_') or "var" in key.split('_'):
                    channels = 1
                else:
                    channels = 3
        
                # Each dart throw is the top left corner of the patch
                ctr = 0
                for dart in darts:
                    not_altered_patch = np.zeros((config.PATCH_WIDTH, config.PATCH_HEIGHT, channels))
                    altered_patch = np.zeros((config.PATCH_WIDTH, config.PATCH_HEIGHT, channels))

                    for x in range(0, config.PATCH_HEIGHT):
                        for y in range(0, config.PATCH_WIDTH):
                            # fill in the patch pixel by pixel

                            # If it's a colour patch, apply the brightness factor
                            if augmentation:
                                if key == "reference_colour" or key == "noisy_colour":
                                    altered_brightness = img_array[dart[0] + x][dart[1] + y] \
                                                       + brightness_factors[i * config.NUM_DARTS + ctr]
                                    altered_brightness[0] = new_data.toColourVal(altered_brightness[0])
                                    altered_brightness[1] = new_data.toColourVal(altered_brightness[1])
                                    altered_brightness[2] = new_data.toColourVal(altered_brightness[2])
    
                                    altered_patch[x][y] = altered_brightness
                                    not_altered_patch[x][y] = img_array[dart[0] + x][dart[1] + y]
                                else:
                                    not_altered_patch[x][y] = img_array[dart[0] + x][dart[1] + y]
                                    altered_patch[x][y] = img_array[dart[0] + x][dart[1] + y]
                            else:
                                not_altered_patch[x][y] = img_array[dart[0] + x][dart[1] + y]

                    # Add the new patch to the array of patches for this key
                    patches[test_or_train][key].append(np.array(not_altered_patch))
                    not_altered_patch = array_to_img(not_altered_patch)
                    not_altered_patch.save("data/patches/" + train_dir + key  + '/' + str(i * config.NUM_DARTS + ctr) + ".png")

                    if augmentation:
                        patches[test_or_train][key].append(np.array(altered_patch))
                        altered_patch = array_to_img(altered_patch)
                        altered_patch.save("data/patches/" + train_dir + "augmented/" + key + '/' + str(i * config.NUM_DARTS + ctr) + ".png")

                    ctr += 1
            full_img_num += 1
    print("Done!")
    return patches 
