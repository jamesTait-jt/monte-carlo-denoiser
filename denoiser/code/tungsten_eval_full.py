from time import time
import math
import sys
import random
import numpy as np
import pickle
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img

import tungsten_data
import config
import weighted_average
import python_weighted_average

def patchify(img, channels, patch_width=config.PATCH_WIDTH, patch_height=config.PATCH_HEIGHT, img_width=1280, img_height=720):
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
    for i in range(img.shape[0]):
        x = patch_height * (i % patches_per_row)
        y = patch_width * (i // patches_per_row) #integer division
        for x1 in range(img[i].shape[0]):
            for y1 in range(img[i].shape[1]):
                img2d[y + y1][x + x1][0] = img[i][y1][x1][0]
                img2d[y + y1][x + x1][1] = img[i][y1][x1][1]
                img2d[y + y1][x + x1][2] = img[i][y1][x1][2]
    return img2d

def normaliseWeights(weights):
    # Normalise weights

    # Subtract by a constant to avoid overflow
    weightmax = np.max(weights, axis=3, keepdims=True)
    weights = weights - weightmax

    weights = np.exp(weights)
    weight_sum = np.sum(weights, axis=3, keepdims=True)
    weights = np.divide(weights, weight_sum)

    return weights


def applyKernel(noisy_img, weights):

    batch_size = 8

    total_patches = weights.shape[0]
    kernel_size = math.sqrt(weights.shape[3])
    kernel_radius = int(math.floor(kernel_size / 2.0))
    paddings = [[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]]
    noisy_img = np.pad(noisy_img, paddings, mode="symmetric")
    weights = normaliseWeights(weights)

    weights_shape = (batch_size, weights.shape[1], weights.shape[2], weights.shape[3])
    noisy_img_shape = (batch_size, noisy_img.shape[1], noisy_img.shape[2], noisy_img.shape[3])

    weights_tensor = tf.placeholder(tf.float32, shape=weights_shape, name="weights")
    noisy_img_tensor = tf.placeholder(tf.float32, shape=noisy_img_shape, name="noisy_img")

    pred = weighted_average.weighted_average(
        noisy_img_tensor,
        weights_tensor
    )

    with tf.Session("") as sess:
        
        denoised_patches = []
        for i in range(0, total_patches, batch_size):
            print(str(i) + "/" + str(total_patches))
            inputs = {
                    weights_tensor : weights[i : i + batch_size],
                    noisy_img_tensor : noisy_img[i : i + batch_size]
            }
            denoised_patches.append(sess.run(pred, feed_dict=inputs))
        
        denoised_patches = np.concatenate(denoised_patches)
        print(denoised_patches.shape)

    return denoised_patches


def getModelInput(index):
    # Extract the model input
    diffuse_or_albedo_div = "diffuse"
    if config.ALBEDO_DIVIDE:
        diffuse_or_albedo_div = "albedo_divided"

    test_in = [
        np.array(patchify(images["test"]["noisy"][diffuse_or_albedo_div][index], 3)),
        np.array(patchify(images["test"]["noisy"][diffuse_or_albedo_div + "_gx"][index], 3)),
        np.array(patchify(images["test"]["noisy"][diffuse_or_albedo_div + "_gy"][index], 3)),
        np.array(patchify(images["test"]["noisy"][diffuse_or_albedo_div + "_var"][index], 1))
    ]

    feature_list = ["normal", "albedo", "depth"]
    for feature in feature_list:
        feature = feature
        # Each feature is split into gradient in X and Y direction, and its
        # corresponding variance
        feature_keys = [feature + "_gx", feature + "_gy", feature + "_var"]
        for key in feature_keys:
            if key.endswith("var") or "depth" in key.split('_'):
                test_in.append(np.array(patchify(images["test"]["noisy"][key][index], 1)))
            else:
                test_in.append(np.array(patchify(images["test"]["noisy"][key][index], 3)))

    return np.concatenate((test_in), 3), test_in[0]

def getDenoisedImg(model_input, noisy_img_patches):
    print("Making prediction... ")
    weights = model.predict(model_input)
    print("Done!")

    print("Applying weights...")
    pred = applyKernel(noisy_img_patches, weights)
    print("Done!")
    pred = weights

    stitched = stitch(pred, config.PATCH_WIDTH, config.PATCH_HEIGHT, 1280, 1280)
    stitched = array_to_img(stitched[0 : 720, 0 : 1280])

    return stitched

def saveAsFigure(stitched, index):
    fig = plt.figure(figsize=(24, 13.5))

    noisy = images["test"]["noisy"]["diffuse"][index]

    noisy_subplot = plt.subplot(211)
    noisy_subplot.set_title("32spp")
    noisy_subplot.imshow(noisy)

    denoised_subplot = plt.subplot(212)
    denoised_subplot.set_title("Denoised")
    denoised_subplot.imshow(stitched)

    save_dir = sys.argv[1].split('/')[1]
    fig.savefig("../data/output/tungsten_denoised{}.png".format(index))


def denoiseTestImage(index):
    # Load in images
    model_input, noisy_img_patches = getModelInput(index)
    stitched = getDenoisedImg(model_input, noisy_img_patches)
    saveAsFigure(stitched, index)

full_pkl_path = "../data/tungsten/pkl/full_dict.pkl"
images = tungsten_data.loadPkl(full_pkl_path)
model = tf.keras.models.load_model(sys.argv[1], compile=False)
index = int(sys.argv[2])

print("\n\n ===== Denoising test image %d ===== " % index)
denoiseTestImage(index)

