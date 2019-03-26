import sys
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
import math

import config
import data
import weighted_average
from denoiser import Denoiser

def patchify(img, channels, patch_width=config.PATCH_WIDTH, patch_height=config.PATCH_HEIGHT, img_width=config.IMAGE_WIDTH, img_height=config.IMAGE_HEIGHT):
    num_patches = (img_width / patch_width) * (img_height / patch_height) 
    patches = np.zeros((int(num_patches), patch_height, patch_width, channels))
    ctr = 0
    for x in range(0, img_height - patch_height + 1, patch_height):
        for y in range(0, img_width - patch_width + 1, patch_width):
            patch = np.zeros((patch_height, patch_width, channels))
            for x1 in range(patch_height):
                for y1 in range(patch_height):
                    patch[x1][y1] = img[x + x1][y + y1]
            #patches[ctr] = array_to_img(patch)
            patches[ctr] = patch
            ctr += 1
    return patches

def stitch(img, patch_width, patch_height, img_width, img_height):
    patches_per_row = int(img_width / patch_width)
    patches_per_col = int(img_height / patch_height)
    img2d = np.zeros((img_height, img_width, 3))
    print(img.shape)
    for i in range(img.shape[0]):
        x = patch_height * (i % patches_per_row)
        y = patch_width * (i // patches_per_row) #integer division
        for x1 in range(img[i].shape[0]):
            for y1 in range(img[i].shape[1]):
                img2d[y + y1][x + x1][0] = img[i][y1][x1][0]
                img2d[y + y1][x + x1][1] = img[i][y1][x1][1]
                img2d[y + y1][x + x1][2] = img[i][y1][x1][2]
    return array_to_img(img2d)

def getFeaturesFromTitle(title):
    all_features = ["sn", "albedo", "depth"]
    title_list = title.split('&')
    title_list[0] = title_list[0].split('/')[1]

    # Filter the list such that only elements from the all_features list are
    # contained
    feature_list = list(filter(lambda x : x in all_features, title_list))
    return feature_list

# Apply the kernel to the noisy test input
def applyKernel(noisy_img, weights):
    kernel_size = math.sqrt(weights.shape[3])
    kernel_radius = int(math.floor(kernel_size / 2.0))
    paddings = tf.constant([[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])
    noisy_img = tf.pad(noisy_img, paddings, mode="SYMMETRIC")
    noisy_img = tf.cast(noisy_img, dtype="float32")

    # Normalise weights
    exp = tf.math.exp(weights)
    weights_sum = tf.reduce_sum(exp, axis=3, keepdims=True)
    weights = tf.divide(exp, weights_sum)

    with tf.Session(""):
        pred = weighted_average.weighted_average(noisy_img, weights).eval()
        return pred

def multiplyAlbedo(predicted_img_patches, noisy_albedo_patches):
    for i in range(len(predicted_img_patches)):
        predicted_img_patches[i] = np.multiply(
            predicted_img_patches[i], noisy_albedo_patches[i] + 0.00316
        )
    return predicted_img_patches

# Load in trained model
model = tf.keras.models.load_model(sys.argv[1], compile=False)

# Load in and process the raw txt files from the renderer
images = data.loadAndPreProcessImages()

# Every model input will have at least a colour buffer (and gradient/variance)
test_in = [
    patchify(np.array(images["test"]["noisy_colour"][0]), 3),
    patchify(np.array(images["test"]["noisy_colour_gradx"][0]), 3),
    patchify(np.array(images["test"]["noisy_colour_grady"][0]), 3),
    patchify(np.array(images["test"]["noisy_colour_var"][0]), 1)
]

# Extract the features that the model has been trained on 
feature_list = getFeaturesFromTitle(sys.argv[1])
feature_list = ["sn", "albedo", "depth"]
for feature in feature_list:
    feature_keys = [feature + "_gradx", feature + "_grady", feature + "_var"]
    for key in feature_keys:
        if key.endswith("var") or "depth" in key.split('_'):
            test_in.append(patchify(images["test"]["noisy_" + key][0], 1))
        else:
            test_in.append(patchify(images["test"]["noisy_" + key][0], 3))


del images

model_input = np.concatenate((test_in), 3)
#noisy_img_patches = test_in[0]
del test_in

weights = model.predict(model_input)
del model_input

pred = applyKernel(noisy_img_patches, weights)

#with open("data/full/test/noisy_albedo_4.txt") as f:
#    noisy_albedo = data.parseFileRGB(f)

#noisy_albedo = patchify(noisy_albedo, 3)
#pred = multiplyAlbedo(pred, noisy_albedo)

stitched = stitch(pred, config.PATCH_WIDTH, config.PATCH_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
save_dir = sys.argv[1].split('/')[1]
stitched.save("data/output/" + save_dir + "denoised.png")
