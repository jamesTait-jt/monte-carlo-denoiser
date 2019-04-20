import math
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import array_to_img


import config
import weighted_average

# Maps from scene index to scene name
scenes_dict = {
    0 : "bathroom",
    1 : "bedroom",
    2 : "living_room"
}

def getModelInput(images, index, feature_list):
    # Extract the model input
    diffuse_or_albedo_div = "diffuse"
    if config.ALBEDO_DIVIDE:
        diffuse_or_albedo_div = "albedo_divided"

    test_in = [
        np.array(images["test"]["noisy"][diffuse_or_albedo_div][index]),
        np.array(images["test"]["noisy"][diffuse_or_albedo_div + "_gx"][index]),
        np.array(images["test"]["noisy"][diffuse_or_albedo_div + "_gy"][index]),
        np.array(images["test"]["noisy"][diffuse_or_albedo_div + "_var"][index])
    ]

    for feature in feature_list:
        feature = feature
        # Each feature is split into gradient in X and Y direction, and its
        # corresponding variance
        feature_keys = [feature + "_gx", feature + "_gy", feature + "_var"]
        for key in feature_keys:
            if key.endswith("var") or "depth" in key.split('_'):
                test_in.append(np.array(images["test"]["noisy"][key][index]))
            else:
                test_in.append(np.array(images["test"]["noisy"][key][index]))

    return np.concatenate((test_in), 2), test_in[0]

# Apply the kernel of weights to an image
def applyKernel(noisy_img, weights):

    # Normalise weights
    def normaliseWeights(weights):

        # Subtract by a constant to avoid overflow
        weightmax = np.max(weights, axis=3, keepdims=True)
        weights = weights - weightmax

        weights = np.exp(weights)
        weight_sum = np.sum(weights, axis=3, keepdims=True)
        weights = np.divide(weights, weight_sum)

        return weights
    
    ########################################################

    total_patches = weights.shape[0]
    kernel_size = math.sqrt(weights.shape[3])
    kernel_radius = int(math.floor(kernel_size / 2.0))
    paddings = [[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]]
    noisy_img = np.pad(noisy_img, paddings, mode="symmetric")
    weights = normaliseWeights(weights)

    weights_shape = (1, weights.shape[1], weights.shape[2], weights.shape[3])
    noisy_img_shape = (1, noisy_img.shape[1], noisy_img.shape[2], noisy_img.shape[3])

    weights_tensor = tf.placeholder(tf.float32, shape=weights_shape, name="weights")
    noisy_img_tensor = tf.placeholder(tf.float32, shape=noisy_img_shape, name="noisy_img")

    pred = weighted_average.weighted_average(
        noisy_img_tensor,
        weights_tensor
    )

    with tf.Session("") as sess:
        
        inputs = {
            weights_tensor : weights,
            noisy_img_tensor : noisy_img
        }
        denoised = sess.run(pred, feed_dict=inputs)

    return denoised

# Multiply back in the noisy albedo and clip
def albedoMultiply(images, pred, index):
    albedo = np.array(images["test"]["noisy"]["albedo"][index])
    albedo = np.expand_dims(albedo, 0)
    pred = pred * (albedo + 0.00316)
    return pred

def saveDictElement(dict_element, save_dir):
    arr = np.array(dict_element)
    arr = arr.clip(0, 1)
    img = array_to_img(arr)
    img.save(save_dir)


# Save the buffers in the correct directory
def saveBuffers(images, save_dir, img_index):
    test_images = images["test"]
    
    # Save the reference diffuse buffer
    saveDictElement(
        test_images["reference"]["diffuse"][img_index],
        save_dir + "reference.png" 
    )

    # Save the reference diffuse buffer after albedo divide
    saveDictElement(
        test_images["reference"]["albedo_divided"][img_index],
        save_dir + "reference_albdiv.png"
    )

    # Save the noisy buffers
    saveDictElement(
        test_images["noisy"]["diffuse"][img_index],
        save_dir + "noisy.png"
    )
    
    saveDictElement(
        test_images["noisy"]["albedo_divided"][img_index],
        save_dir + "noisy_albdiv.png"
    )

    saveDictElement(
        test_images["noisy"]["depth"][img_index],
        save_dir + "noisy_depth.png"
    )

    saveDictElement(
        test_images["noisy"]["albedo"][img_index],
        save_dir + "noisy_albedo.png"
    )

    saveDictElement(
        test_images["noisy"]["normal"][img_index],
        save_dir + "noisy_normal.png"
    )
