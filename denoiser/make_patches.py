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

def makePatches():
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
        }
    }

    print("Generating patches...")
    full_images = new_data.loadAndPreProcessImages()
    for test_or_train in full_images:
        
        if (test_or_train == "train"):
            train_dir = "train/"
            num_scenes = config.TRAIN_SCENES
        else:
            train_dir = "test/"
            num_scenes = config.TEST_SCENES

        for key in full_images[test_or_train]:
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
                    ctr += 1
                    new_patch = np.zeros((config.PATCH_WIDTH, config.PATCH_HEIGHT, channels))
                    for x in range(0, config.PATCH_HEIGHT):
                        for y in range(0, config.PATCH_WIDTH):
                            # fill in the patch pixel by pixel
                            new_patch[x][y] = img_array[dart[0] + x][dart[1] + y]
                    # Add the new patch to the array of patches for this key
                    patches[test_or_train][key].append(np.array(new_patch))
                    #new_patch = array_to_img(new_patch)
                    #new_patch.save(config.PATCH_SAVE_DIRS[i] + str(ctr) + ".png")
    print("Done!")
    return patches 
