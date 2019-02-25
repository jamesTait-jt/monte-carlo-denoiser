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

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

reference_dir = "data/reference"
noisy_dir = "data/noisy"

reference_data = []
for img in sorted(os.listdir(reference_dir)):
    img = load_img(reference_dir + "/" + img)
    img = img_to_array(img)
    #img = img.reshape((1,) + img.shape)
    reference_data.append(img)

reference_data = np.array(reference_data) / 255.

noisy_data = []
for img in sorted(os.listdir(noisy_dir)):
    img = load_img(noisy_dir + "/" + img)
    img = img_to_array(img)
    #img = img.reshape((1,) + img.shape)
    noisy_data.append(img)

noisy_data = np.array(noisy_data) / 255.

reference_train = reference_data[int(reference_data.shape[0] * 0.20) :]
reference_test  = reference_data[: int(reference_data.shape[0] * 0.20)]

noisy_train = noisy_data[int(noisy_data.shape[0] * 0.20) :]
noisy_test  = noisy_data[: int(noisy_data.shape[0] * 0.20)]

data = {
    "train" : {
        "colour" : {
            "reference" : reference_train,
            "noisy" : noisy_train
        }
    },
    "test" : {
        "colour" : {
            "reference" : reference_test,
            "noisy" : noisy_test
        }
    }
}

