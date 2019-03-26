import random
import numpy as np
from scipy import ndimage

from keras.preprocessing.image import array_to_img, img_to_array, load_img

import config

# Clips a float between 0 and 1
def toColourVal(x):
    x = float(x)
    if x < 0:
        x = 0
    elif x > 1:
        x = 1
    return x

def luminance_img(img):
    luminance_img = np.zeros((img.shape[0], img.shape[1], 1))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            luminance_img[i][j][0] = luminance(img[i][j])
    return luminance_img

def luminance(rgb):
    return 0.299 * pow(rgb[0], 2) + \
           0.587 * pow(rgb[1], 2) + \
           0.144 * pow(rgb[2], 2)

# Parses a .txt file of vec3 to an rgb numpy array
def parseFileRGB(f):
    data = np.array(
        [[toColourVal(x.split(' ')[0]),
          toColourVal(x.split(' ')[1]),
          toColourVal(x.split(' ')[2])] for x in f.read().split(',')[:-1]]
    )
    data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    return data

# Parses a .txt file of greyscale colour values to a greyscale numpy array
def parseFileGreyscale(f):
    data = np.array([float(x) for x in f.read().split(',')[:-1]])
    data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))
    return data


def preProcessReferenceColour(is_train):
    if is_train:
        train_dir = "train/"
        num_scenes = config.TRAIN_SCENES
    else:
        train_dir = "test/"
        num_scenes = config.TEST_SCENES

    print("Loading in reference colour...")
    colour_data_arr = []
    for i in range(num_scenes):

        if not is_train:
            j = i + config.TRAIN_SCENES
        else:
            j = i

        with open("data/full/" + train_dir + "reference_colour_" + str(j) + ".txt") as f:
            data = parseFileRGB(f)
            array_to_img(data).save("data/full/reference_images/" + str(j) + ".png")

        with open("data/full/" + train_dir + "reference_albedo_" + str(j) +".txt") as f:
            albedo_data = parseFileRGB(f)

            factored_colour = np.clip(np.divide(data, albedo_data + 0.00316), 0, 1)
            colour_data_arr.append(factored_colour)

    print("Done!")
    return colour_data_arr

def preProcessNoisyColour(is_train):
    if is_train:
        train_dir = "train/"
        num_scenes = config.TRAIN_SCENES
    else:
        train_dir = "test/"
        num_scenes = config.TEST_SCENES

    print("Loading in noisy colour...")
    colour_data_arr = []
    gradx_arr = []
    grady_arr = []
    var_arr = []
    for i in range(num_scenes):

        if not is_train:
            j = i + config.TRAIN_SCENES
        else:
            j = i

        with open("data/full/" + train_dir + "noisy_colour_" + str(j) + ".txt") as f:
            colour_data = parseFileRGB(f)

        with open("data/full/" + train_dir + "noisy_colour_vars_" + str(j) + ".txt") as f:
            var_data = parseFileRGB(f)

        with open("data/full/" + train_dir + "noisy_albedo_" + str(j) +".txt") as f2:
            albedo_data = parseFileRGB(f2)

            factored_colour = np.clip(np.divide(colour_data, albedo_data + 0.00316), 0, 1)
            factored_var = np.divide(var_data, pow(albedo_data + 0.00316, 2))
            factored_var = luminance_img(factored_var)
            factored_var = factored_var / np.amax(factored_var)

            colour_data_arr.append(factored_colour)
            var_arr.append(factored_var)
            img = array_to_img(factored_colour)

        gradx_arr.append(ndimage.sobel(img, axis=0, mode='constant') / 255.0)
        grady_arr.append(ndimage.sobel(img, axis=1, mode='constant') / 255.0)


    print("Done!")
    return colour_data_arr, gradx_arr, grady_arr, var_arr

def preProcessAlbedo(is_train):
    if is_train:
        train_dir = "train/"
        num_scenes = config.TRAIN_SCENES
    else:
        train_dir = "test/"
        num_scenes = config.TEST_SCENES

    print("Loading in albedo...")
    gradx_arr = []
    grady_arr = []
    var_arr = []
    for i in range(num_scenes):

        if not is_train:
            j = i + config.TRAIN_SCENES
        else:
            j = i

        with open("data/full/" + train_dir + "noisy_albedo_" + str(j) + ".txt") as f:
            data = parseFileRGB(f)

            img = array_to_img(data)
            gradx = ndimage.sobel(img, axis=0, mode='constant') / 255.0
            grady = ndimage.sobel(img, axis=1, mode='constant') / 255.0
            gradx_arr.append(gradx)
            grady_arr.append(grady)

        with open("data/full/" + train_dir + "noisy_albedo_vars_" + str(j) + ".txt") as f:
            var_data = parseFileRGB(f)
            var_data = luminance_img(var_data)
            var_data = var_data / np.amax(var_data)
            var_arr.append(var_data)

    print("Done!")
    return gradx_arr, grady_arr, var_arr

# The depths are normalised between 0 and 1
def preProcessDepth(is_train):
    if is_train:
        train_dir = "train/"
        num_scenes = config.TRAIN_SCENES
    else:
        train_dir = "test/"
        num_scenes = config.TEST_SCENES

    print("Loading in depth...")
    gradx_arr = []
    grady_arr = []
    var_arr = []
    for i in range(num_scenes):

        if not is_train:
            j = i + config.TRAIN_SCENES
        else:
            j = i

        with open("data/full/" + train_dir + "noisy_depth_" + str(j) + ".txt") as f:
            data = parseFileGreyscale(f)
            data /= np.amax(data)

            img = array_to_img(data)
            gradx = ndimage.sobel(img, axis=0, mode='constant')
            grady = ndimage.sobel(img, axis=1, mode='constant')

            gradx = np.reshape(gradx, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)) / 255.0
            grady = np.reshape(grady, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)) / 255.0
            gradx_arr.append(gradx)
            grady_arr.append(grady)

        with open("data/full/" + train_dir + "noisy_depth_vars_" + str(j) + ".txt") as f:
            var_data = parseFileGreyscale(f)
            var_data = var_data / np.amax(var_data)
            var_arr.append(var_data)

    print("Done!")
    return gradx_arr, grady_arr, var_arr

# Nothing special is done to the surface normals
def preProcessSurfaceNormal(is_train):
    if is_train:
        train_dir = "train/"
        num_scenes = config.TRAIN_SCENES
    else:
        train_dir = "test/"
        num_scenes = config.TEST_SCENES

    print("Loading in surface normals...")
    gradx_arr = []
    grady_arr = []
    var_arr = []
    for i in range(num_scenes):

        if not is_train:
            j = i + config.TRAIN_SCENES
        else:
            j = i

        with open("data/full/" + train_dir + "noisy_sn_" + str(j) + ".txt") as f:
            data = parseFileRGB(f)

            img = array_to_img(data)
            gradx = ndimage.sobel(img, axis=0, mode='constant') / 255.0
            grady = ndimage.sobel(img, axis=1, mode='constant') / 255.0
            gradx_arr.append(gradx)
            grady_arr.append(grady)

        with open("data/full/" + train_dir + "noisy_sn_vars_" + str(j) + ".txt") as f:
            var_data = parseFileRGB(f)
            var_data = luminance_img(var_data)
            var_data = var_data / np.amax(var_data)
            var_arr.append(var_data)

    print("Done!")
    return gradx_arr, grady_arr, var_arr

def saveImages(images):
    print("Saving images...")
    for test_or_train in images:

        if test_or_train == "train":
            train_dir = "train/"
            num_scenes = config.TRAIN_SCENES
        else:
            train_dir = "test/"
            num_scenes = config.TEST_SCENES

        for key in images[test_or_train]:
            print("Saving " + test_or_train + ": " + key)
            for i in range(num_scenes):
                img = array_to_img(images[test_or_train][key][i])
                img.save("data/full/" + train_dir + key + "_" + str(i) +  ".png")

def loadAndPreProcessImages():
    train_noisy_colour, \
    train_noisy_colour_gradx, \
    train_noisy_colour_grady, \
    train_noisy_colour_var = preProcessNoisyColour(is_train=True)

    train_sn_gradx, \
    train_sn_grady, \
    train_sn_var = preProcessSurfaceNormal(is_train=True)

    train_albedo_gradx, \
    train_albedo_grady, \
    train_albedo_var = preProcessAlbedo(is_train=True)

    train_depth_gradx, \
    train_depth_grady, \
    train_depth_var = preProcessDepth(is_train=True)

    test_noisy_colour, \
    test_noisy_colour_gradx, \
    test_noisy_colour_grady, \
    test_noisy_colour_var = preProcessNoisyColour(is_train=False)

    test_sn_gradx, \
    test_sn_grady, \
    test_sn_var = preProcessSurfaceNormal(is_train=False)

    test_albedo_gradx, \
    test_albedo_grady, \
    test_albedo_var = preProcessAlbedo(is_train=False)

    test_depth_gradx, \
    test_depth_grady, \
    test_depth_var = preProcessDepth(is_train=False)

    full_images = {
        "train" : {
            "reference_colour" : preProcessReferenceColour(is_train=True),
            "noisy_colour" : train_noisy_colour,
            "noisy_colour_gradx" : train_noisy_colour_gradx,
            "noisy_colour_grady" : train_noisy_colour_grady,
            "noisy_colour_var" : train_noisy_colour_var,
            "noisy_sn_gradx" : train_sn_gradx,
            "noisy_sn_grady" : train_sn_grady,
            "noisy_sn_var" : train_sn_var,
            "noisy_albedo_gradx" : train_albedo_gradx,
            "noisy_albedo_grady" : train_albedo_grady,
            "noisy_albedo_var" : train_albedo_var,
            "noisy_depth_gradx" : train_depth_gradx,
            "noisy_depth_grady" : train_depth_grady,
            "noisy_depth_var" : train_depth_var
        },

        "test" : {
            "reference_colour" : preProcessReferenceColour(is_train=False),
            "noisy_colour" : test_noisy_colour,
            "noisy_colour_gradx" : test_noisy_colour_gradx,
            "noisy_colour_grady" : test_noisy_colour_grady,
            "noisy_colour_var" : test_noisy_colour_var,
            "noisy_sn_gradx" : test_sn_gradx,
            "noisy_sn_grady" : test_sn_grady,
            "noisy_sn_var" : test_sn_var,
            "noisy_albedo_gradx" : test_albedo_gradx,
            "noisy_albedo_grady" : test_albedo_grady,
            "noisy_albedo_var" : test_albedo_var,
            "noisy_depth_gradx" : test_depth_gradx,
            "noisy_depth_grady" : test_depth_grady,
            "noisy_depth_var" : test_depth_var
        }
    }
    saveImages(full_images)
    return full_images

# Generate the "darts" that we will throw at the images in order to select the patches
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

# Gets a list of random floats in (-limit, limit) to apply to reference and
# noisy colour patches
def generateBrightnessFactors(limit):
    factors = [random.uniform(-limit, limit) for i in range(config.TRAIN_SCENES * config.NUM_DARTS)]
    return factors

# Initialise an empty dictionary to store the image patches in
def initialisePatchDict():
    return {
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

# If the buffer is a variance/depth buffer then we only have one channel
def setNumChannels(key):
    if "depth" in key.split('_') or "var" in key.split('_'):
        return 1
    else:
        return 3

def throwDart(
    dart,
    key,
    img_array,
    channels,
    ctr,
    train_dir,
    test_or_train,
    scene_num,
    patches,
    brightness_factors
):
    not_altered_patch = np.zeros((config.PATCH_WIDTH, config.PATCH_HEIGHT, channels))
    altered_patch = np.zeros((config.PATCH_WIDTH, config.PATCH_HEIGHT, channels))

    for x in range(0, config.PATCH_HEIGHT):
        for y in range(0, config.PATCH_WIDTH):

            # If it's a colour patch, apply the brightness factor
            if test_or_train == "train":
                if key in ["reference_colour", "noisy_colour"]:
                    altered_brightness = img_array[dart[0] + x][dart[1] + y] \
                                       + brightness_factors[scene_num * config.NUM_DARTS + ctr]
                    altered_brightness[0] = toColourVal(altered_brightness[0])
                    altered_brightness[1] = toColourVal(altered_brightness[1])
                    altered_brightness[2] = toColourVal(altered_brightness[2])

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
    not_altered_patch.save(
        "data/patches/" + train_dir + key  + '/' + str(scene_num * config.NUM_DARTS + ctr) + ".png"
    )

    if test_or_train == "train":
        patches[test_or_train][key].append(np.array(altered_patch))
        altered_patch = array_to_img(altered_patch)
        altered_patch.save(
            "data/patches/" + train_dir + "augmented/" + key + '/' + str(scene_num * config.NUM_DARTS + ctr) + ".png"
        )


def makePatches(seed):

    patches = initialisePatchDict()
    
    if config.MAKE_NEW_PATCHES:
        random.seed(seed)
        darts = generate_darts(
            config.NUM_DARTS,
            config.IMAGE_WIDTH,
            config.IMAGE_HEIGHT,
            config.PATCH_WIDTH,
            config.PATCH_HEIGHT
        )


        full_images = loadAndPreProcessImages()
        brightness_factors = generateBrightnessFactors(0.3)

        print("Generating patches...")
        for test_or_train in full_images:
            if test_or_train == "train":
                train_dir = "train/"
                num_scenes = config.TRAIN_SCENES
            else:
                train_dir = "test/"
                num_scenes = config.TEST_SCENES

            full_img_num = 0
            for key in full_images[test_or_train]:
                for i in range(num_scenes):
                    img_array = full_images[test_or_train][key][i]
                    channels = setNumChannels(key)

                    # Each dart throw is the top left corner of the patch
                    ctr = 0
                    for dart in darts:
                        throwDart(
                            dart,
                            key,
                            img_array,
                            channels,
                            ctr,
                            train_dir,
                            test_or_train,
                            i,
                            patches,
                            brightness_factors
                        )
                        ctr += 1

                full_img_num += 1
        print("Done!")

    else:
        print("Not generating new patches. Loading in patches...")
        for test_or_train in patches:

            if test_or_train == "train":
                num_scenes = config.TRAIN_SCENES
            else:
                num_scenes = config.TEST_SCENES

            for key in patches[test_or_train]:
                for i in range(num_scenes * config.NUM_DARTS):
                    img_path = "data/patches/" + test_or_train + "/" + key + "/" + str(i) + ".png"
                    img = load_img(img_path)
                    img = img_to_array(img) / 255.0
                    if "var" in key or "depth" in key:
                        img = np.mean(img, axis=2, keepdims=True)
                    patches[test_or_train][key].append(img)

                    if config.AUGMENTATION:
                        if test_or_train == "train":
                            augmented_img_path = "data/patches/" + test_or_train + "/augmented/" + key + "/" + str(i) + ".png"
                            img = load_img(augmented_img_path)
                            img = img_to_array(img) / 255.0
                            if "var" in key or "depth" in key:
                                img = np.mean(img, axis=2, keepdims=True)
                            patches[test_or_train][key].append(img)

    return patches
