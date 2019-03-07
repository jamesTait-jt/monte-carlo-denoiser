from PIL import Image
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import config

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
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

data_list = []
for i in range(len(config.PATCH_SAVE_DIRS)):
    # Extract the image for storing in dict
    img_dir = config.PATCH_SAVE_DIRS[i]

    data = []
    for img in sorted(os.listdir(img_dir)):
        img = load_img(img_dir + img)
        img = img_to_array(img)
        data.append(img)
    data = np.array(data) / 255.0

    # Split into training and test data (80:20 split)
    train = data[int(data.shape[0] * 0.20) :] 
    test = data[: int(data.shape[0] * 0.20)] 

    data_list.append((train, test))

# Define the data dictionary
data = {
    "train" : {
        "colour" : {
            "reference" : data_list[0][0],
            "noisy" : data_list[1][0]
        },
        "colour_gradx" : {
            "noisy" : data_list[2][0]
        },
        "colour_grady" : {
            "noisy" : data_list[3][0]
        },
        "colour_var" : {
            "noisy" : data_list[4][0]
        },
        "sn_gradx" : {
            "noisy" : data_list[5][0]
        },
        "sn_grady" : {
            "noisy" : data_list[6][0]
        },
        "sn_var" : {
            "noisy" : data_list[7][0]
        },
        "albedo_gradx" : {
            "noisy" : data_list[8][0]
        },
        "albedo_grady" : {
            "noisy" : data_list[9][0]
        },
        "albedo_var" : {
            "noisy" : data_list[10][0]
        },
        "depth_gradx" : {
            "noisy" : data_list[11][0]
        },
        "depth_grady" : {
            "noisy" : data_list[12][0]
        },
        "depth_var" : {
            "noisy" : data_list[13][0]
        }
    },
    "test" : {
        "colour" : {
            "reference" : data_list[0][1],
            "noisy" : data_list[1][1]
        },
        "colour_gradx" : {
            "noisy" : data_list[2][1]
        },
        "colour_grady" : {
            "noisy" : data_list[3][1]
        },
        "colour_var" : {
            "noisy" : data_list[4][1]
        },
        "sn_gradx" : {
            "noisy" : data_list[5][1]
        },
        "sn_grady" : {
            "noisy" : data_list[6][1]
        },
        "sn_var" : {
            "noisy" : data_list[7][1]
        },
        "albedo_gradx" : {
            "noisy" : data_list[8][1]
        },
        "albedo_grady" : {
            "noisy" : data_list[9][1]
        },
        "albedo_var" : {
            "noisy" : data_list[10][1]
        },
        "depth_gradx" : {
            "noisy" : data_list[11][1]
        },
        "depth_grady" : {
            "noisy" : data_list[12][1]
        },
        "depth_var" : {
            "noisy" : data_list[13][1]
        }
    }
}
