import numpy as np
import config
from scipy import ndimage

from keras.preprocessing.image import array_to_img, img_to_array, load_img

def toColourVal(x):
    x = float(x)
    if x < 0:
        x = 0
    elif x > 1:
        x = 1
    return x

def preProcessReferenceColour(is_train):

    if (is_train):
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
            data = np.array([[toColourVal(x.split(' ')[0]), toColourVal(x.split(' ')[1]), toColourVal(x.split(' ')[2])] for x in f.read().split(',')[:-1]])
            data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
            colour_data_arr.append(data)
    print("Done!")
    return colour_data_arr

def preProcessNoisyColour(is_train):

    if (is_train):
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
            colour_data = np.array([[toColourVal(x.split(' ')[0]), toColourVal(x.split(' ')[1]), toColourVal(x.split(' ')[2])] for x in f.read().split(',')[:-1]])
            colour_data = np.reshape(colour_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
            colour_data_arr.append(colour_data)

            img = array_to_img(colour_data)        
            gradx_arr.append(ndimage.sobel(img, axis=0, mode='constant') / 255.0)
            grady_arr.append(ndimage.sobel(img, axis=1, mode='constant') / 255.0)

        with open("data/full/" + train_dir + "noisy_colour_vars_" + str(j) + ".txt") as f:
            var_data = np.array([float(x) for x in f.read().split(',')[:-1]])
            var_data = np.reshape(var_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))
            var_arr.append(var_data / np.amax(var_data))

    print("Done!")
    return colour_data_arr, gradx_arr, grady_arr, var_arr

def preProcessAlbedo(is_train):

    if (is_train):
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
            data = np.array([[float(x.split(' ')[0]), float(x.split(' ')[1]), float(x.split(' ')[2])] for x in (f.read().split(',')[:-1])])
            data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))

            img = array_to_img(data)        
            gradx = ndimage.sobel(img, axis=0, mode='constant') / 255.0
            grady = ndimage.sobel(img, axis=1, mode='constant') / 255.0
            gradx_arr.append(gradx)
            grady_arr.append(grady)

        with open("data/full/" + train_dir + "noisy_albedo_vars_" + str(j) + ".txt") as f:
            var_data = np.array([float(x) for x in f.read().split(',')[:-1]])
            var_data = np.reshape(var_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))
            var_data = var_data / np.amax(var_data)
            var_arr.append(var_data)

    print("Done!")
    return gradx_arr, grady_arr, var_arr

# The depths are normalised between 0 and 1
def preProcessDepth(is_train):

    if (is_train):
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
            data = np.array([float(x) for x in f.read().split(',')[:-1]])
            data /= np.max(data)
            data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))

            img = array_to_img(data)        
            gradx = ndimage.sobel(img, axis=0, mode='constant')
            grady = ndimage.sobel(img, axis=1, mode='constant')

            gradx = np.reshape(gradx, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)) / 255.0
            grady = np.reshape(grady, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)) / 255.0
            gradx_arr.append(gradx)
            grady_arr.append(grady)

        with open("data/full/" + train_dir + "noisy_depth_vars_" + str(j) + ".txt") as f:
            var_data = np.array([float(x) for x in f.read().split(',')[:-1]])
            var_data = np.reshape(var_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))
            var_data = var_data / np.amax(var_data)
            var_arr.append(var_data)

    print("Done!")
    return gradx_arr, grady_arr, var_arr

# Nothing special is done to the surface normals
def preProcessSurfaceNormal(is_train):

    if (is_train):
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
            data = np.array([[float(x.split(' ')[0]), float(x.split(' ')[1]), float(x.split(' ')[2])] for x in (f.read().split(',')[:-1])])
            data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))

            img = array_to_img(data)        
            gradx = ndimage.sobel(img, axis=0, mode='constant') / 255.0
            grady = ndimage.sobel(img, axis=1, mode='constant') / 255.0
            gradx_arr.append(gradx)
            grady_arr.append(grady)

        with open("data/full/" + train_dir + "noisy_sn_vars_" + str(j) + ".txt") as f:
            var_data = np.array([float(x) for x in f.read().split(',')[:-1]])
            var_data = np.reshape(var_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))
            var_data = var_data / np.amax(var_data)
            var_arr.append(var_data)

    print("Done!")
    return gradx_arr, grady_arr, var_arr

def saveImages(images):
    print("Saving images...")
    for test_or_train in images:

        if (test_or_train == "train"):
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
    train_noisy_colour, train_noisy_colour_gradx, train_noisy_colour_grady, train_noisy_colour_var = preProcessNoisyColour(True)
    train_sn_gradx, train_sn_grady, train_sn_var = preProcessSurfaceNormal(True)
    train_albedo_gradx, train_albedo_grady, train_albedo_var = preProcessAlbedo(True)
    train_depth_gradx, train_depth_grady, train_depth_var = preProcessDepth(True)
    
    test_noisy_colour, test_noisy_colour_gradx, test_noisy_colour_grady, test_noisy_colour_var = preProcessNoisyColour(False)
    test_sn_gradx, test_sn_grady, test_sn_var = preProcessSurfaceNormal(False)
    test_albedo_gradx, test_albedo_grady, test_albedo_var = preProcessAlbedo(False)
    test_depth_gradx, test_depth_grady, test_depth_var = preProcessDepth(False)

    full_images = {
    
        "train" : {
            "reference_colour" : preProcessReferenceColour(True),
            "noisy_colour" : train_noisy_colour,
            "noisy_colour_gradx" : train_noisy_colour_gradx,
            "noisy_colour_grady" : train_noisy_colour_grady,
            "noisy_colour_var" : train_noisy_colour_var,
            "noisy_sn_gradx" : train_sn_gradx,
            "noisy_sn_grady" : train_sn_gradx,
            "noisy_sn_var" : train_sn_var,
            "noisy_albedo_gradx" : train_albedo_gradx,
            "noisy_albedo_grady" : train_albedo_gradx,
            "noisy_albedo_var" : train_albedo_var,
            "noisy_depth_gradx" : train_depth_gradx,
            "noisy_depth_grady" : train_depth_gradx,
            "noisy_depth_var" : train_depth_var
        },

        "test" : {
            "reference_colour" : preProcessReferenceColour(False),
            "noisy_colour" : test_noisy_colour,
            "noisy_colour_gradx" : test_noisy_colour_gradx,
            "noisy_colour_grady" : test_noisy_colour_grady,
            "noisy_colour_var" : test_noisy_colour_var,
            "noisy_sn_gradx" : test_sn_gradx,
            "noisy_sn_grady" : test_sn_gradx,
            "noisy_sn_var" : test_sn_var,
            "noisy_albedo_gradx" : test_albedo_gradx,
            "noisy_albedo_grady" : test_albedo_gradx,
            "noisy_albedo_var" : test_albedo_var,
            "noisy_depth_gradx" : test_depth_gradx,
            "noisy_depth_grady" : test_depth_gradx,
            "noisy_depth_var" : test_depth_var
        }
    }
    saveImages(full_images)
    return full_images

