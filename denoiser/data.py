from PIL import Image
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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

# List of directories for various data
IMG_DIRS = [
    "data/patches/reference_colour/",
    "data/patches/noisy_colour/",

    "data/patches/reference_colour_gradx/",
    "data/patches/noisy_colour_gradx/",

    "data/patches/reference_colour_grady/",
    "data/patches/noisy_colour_grady/",

    "data/patches/reference_sn/",
    "data/patches/noisy_sn/",

    "data/patches/reference_sn_gradx/",
    "data/patches/noisy_sn_gradx/",

    "data/patches/reference_sn_grady/",
    "data/patches/noisy_sn_grady/",

    "data/patches/reference_albedo/",
    "data/patches/noisy_albedo/",

    "data/patches/reference_albedo_gradx/",
    "data/patches/noisy_albedo_gradx/",

    "data/patches/reference_albedo_grady/",
    "data/patches/noisy_albedo_grady/",

    "data/patches/reference_depth/",
    "data/patches/noisy_depth/",

    "data/patches/reference_depth_gradx/",
    "data/patches/noisy_depth_gradx/",

    "data/patches/reference_depth_grady/",
    "data/patches/noisy_depth_grady/",

    "data/patches/reference_colour_vars/",
    "data/patches/noisy_colour_vars/",

    "data/patches/reference_sn_vars/",
    "data/patches/noisy_sn_vars/",

    "data/patches/reference_albedo_vars/",
    "data/patches/noisy_albedo_vars/",

    "data/patches/reference_depth_vars/",
    "data/patches/noisy_depth_vars/"
]

data_list = []
for i in range(len(IMG_DIRS)):
    # Extract the image for storing in dict
    img_dir = IMG_DIRS[i]

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
            "reference" : data_list[2][0],
            "noisy" : data_list[3][0]
        },
        "colour_grady" : {
            "reference" : data_list[4][0],
            "noisy" : data_list[5][0]
        },
        "sn" : {
            "reference" : data_list[6][0],
            "noisy" : data_list[7][0]
        },
        "sn_gradx" : {
            "reference" : data_list[8][0],
            "noisy" : data_list[9][0]
        },
        "sn_grady" : {
            "reference" : data_list[10][0],
            "noisy" : data_list[11][0]
        },
        "albedo" : {
            "reference" : data_list[12][0],
            "noisy" : data_list[13][0]
        },
        "albedo_gradx" : {
            "reference" : data_list[14][0],
            "noisy" : data_list[15][0]
        },
        "albedo_grady" : {
            "reference" : data_list[16][0],
            "noisy" : data_list[17][0]
        },
        "depth" : {
            "reference" : convert_channels_3_to_1(data_list[18][0]),
            "noisy" : convert_channels_3_to_1(data_list[19][0])
        },
        "depth_gradx" : {
            "reference" : convert_channels_3_to_1(data_list[20][0]),
            "noisy" : convert_channels_3_to_1(data_list[21][0])
        },
        "depth_grady" : {
            "reference" : convert_channels_3_to_1(data_list[22][0]),
            "noisy" : convert_channels_3_to_1(data_list[23][0])
        },
        "colour_var" : {
            "reference" : convert_channels_3_to_1(data_list[24][0]),
            "noisy" : convert_channels_3_to_1(data_list[25][0])
        },
        "sn_var" : {
            "reference" : convert_channels_3_to_1(data_list[26][0]),
            "noisy" : convert_channels_3_to_1(data_list[27][0])
        },
        "albedo_var" : {
            "reference" : convert_channels_3_to_1(data_list[28][0]),
            "noisy" : convert_channels_3_to_1(data_list[29][0])
        },
        "depth_var" : {
            "reference" : convert_channels_3_to_1(data_list[30][0]),
            "noisy" : convert_channels_3_to_1(data_list[31][0])
        }
    },
    "test" : {
        "colour" : {
            "reference" : data_list[0][1],
            "noisy" : data_list[1][1]
        },
        "colour_gradx" : {
            "reference" : data_list[2][1],
            "noisy" : data_list[3][1]
        },
        "colour_grady" : {
            "reference" : data_list[4][1],
            "noisy" : data_list[5][1]
        },
        "sn" : {
            "reference" : data_list[6][1],
            "noisy" : data_list[7][1]
        },
        "sn_gradx" : {
            "reference" : data_list[8][1],
            "noisy" : data_list[9][1]
        },
        "sn_grady" : {
            "reference" : data_list[10][1],
            "noisy" : data_list[11][1]
        },
        "albedo" : {
            "reference" : data_list[12][1],
            "noisy" : data_list[13][1]
        },
        "albedo_gradx" : {
            "reference" : data_list[14][1],
            "noisy" : data_list[15][1]
        },
        "albedo_grady" : {
            "reference" : data_list[16][1],
            "noisy" : data_list[17][1]
        },
        "depth" : {
            "reference" : convert_channels_3_to_1(data_list[18][1]),
            "noisy" : convert_channels_3_to_1(data_list[19][1])
        },
        "depth_gradx" : {
            "reference" : convert_channels_3_to_1(data_list[20][1]),
            "noisy" : convert_channels_3_to_1(data_list[21][1])
        },
        "depth_grady" : {
            "reference" : convert_channels_3_to_1(data_list[22][1]),
            "noisy" : convert_channels_3_to_1(data_list[23][1])
        },
        "colour_var" : {
            "reference" : convert_channels_3_to_1(data_list[24][1]),
            "noisy" : convert_channels_3_to_1(data_list[25][1])
        },
        "sn_var" : {
            "reference" : convert_channels_3_to_1(data_list[26][1]),
            "noisy" : convert_channels_3_to_1(data_list[27][1])
        },
        "albedo_var" : {
            "reference" : convert_channels_3_to_1(data_list[28][1]),
            "noisy" : convert_channels_3_to_1(data_list[29][1])
        },
        "depth_var" : {
            "reference" : convert_channels_3_to_1(data_list[30][1]),
            "noisy" : convert_channels_3_to_1(data_list[31][1])
        }
    }
}
