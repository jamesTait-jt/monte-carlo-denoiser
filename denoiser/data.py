from PIL import Image
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import config
import make_patches

# Shuffle two arrays in the same way so that they keep their correspondance
def shuffle_two_arrays(a, b):
    s = np.arange(0, len(a), 1)
    np.random.shuffle(s)
    new_a = np.zeros((226, 64, 64, 3))
    new_b = np.zeros((226, 64, 64, 3))
    for i in range(len(a)):
        new_a[i] = a[s[i]]
        new_b[i] = s[s[i]]
    return new_a, new_b

# Takes a list of AxBxCx3 array and output AxBxCx1 where the value is the first
# dimension of the 3D vector
def convert_channels_3_to_1(data):
    shape = data.shape
    new_data = np.zeros((shape[0], shape[1], shape[2], 1))
    for i in range(len(data)):
        for x in range(shape[1]):
            for y in range(shape[2]):
                new_data[i][x][y] = data[i][x][y][0]
    return new_data

def convert_channels_7_to_3(data):
    shape = data.shape
    new_data = np.zeros((shape[0], shape[1], shape[2], 3))
    for i in range(len(data)):
        for x in range(shape[1]):
            for y in range(shape[2]):
                new_data[i][x][y][0] = data[i][x][y][0]
                new_data[i][x][y][1] = data[i][x][y][1]
                new_data[i][x][y][2] = data[i][x][y][2]
    return new_data

datagen = ImageDataGenerator(
    rotation_range=40,
    horizontal_flip=True,
    fill_mode='nearest'
)


data = {
    "train" : {
        "colour" : {
            "reference" : None,
            "noisy" : None
        },
        "colour_gradx" : {
            "noisy" : None
        },
        "colour_grady" : {
            "noisy" : None
        },
        "colour_var" : {
            "noisy" : None
        },
        "sn_gradx" : {
            "noisy" : None
        },
        "sn_grady" : {
            "noisy" : None
        },
        "sn_var" : {
            "noisy" : None
        },
        "albedo_gradx" : {
            "noisy" : None
        },
        "albedo_grady" : {
            "noisy" : None
        },
        "albedo_var" : {
            "noisy" : None
        },
        "depth_gradx" : {
            "noisy" : None
        },
        "depth_grady" : {
            "noisy" : None
        },
        "depth_var" : {
            "noisy" : None
        }
    },
    "test" : {
        "colour" : {
            "reference" : None,
            "noisy" : None
        },
        "colour_gradx" : {
            "noisy" : None
        },
        "colour_grady" : {
            "noisy" : None
        },
        "colour_var" : {
            "noisy" : None
        },
        "sn_gradx" : {
            "noisy" : None
        },
        "sn_grady" : {
            "noisy" : None
        },
        "sn_var" : {
            "noisy" : None
        },
        "albedo_gradx" : {
            "noisy" : None
        },
        "albedo_grady" : {
            "noisy" : None
        },
        "albedo_var" : {
            "noisy" : None
        },
        "depth_gradx" : {
            "noisy" : None
        },
        "depth_grady" : {
            "noisy" : None
        },
        "depth_var" : {
            "noisy" : None
        }
    }
}

data_list = []
for key in make_patches.patches:
    for i in range(config.TOTAL_SCENES):
        patches = np.array(make_patches.patches[key])
        train = patches[int(patches.shape[0] * 0.20) :]
        test = patches[: int(patches.shape[0] * 0.20) ]
        data_list.append((train, test))

        key_list = key.split('_')
        if (len(key_list) == 3):
            data["train"][key_list[1] + '_' + key_list[2]][key_list[0]] = train
            data["test"][key_list[1] + '_' + key_list[2]][key_list[0]] = test
        else:
            data["train"][key_list[1]][key_list[0]] = train
            data["test"][key_list[1]][key_list[0]] = test
