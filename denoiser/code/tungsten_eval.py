import math
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img

import tungsten_data
import config
import weighted_average

def applyKernel(noisy_img, weights):
    kernel_size = math.sqrt(weights.shape[3])
    kernel_radius = int(math.floor(kernel_size / 2.0))
    paddings = tf.constant([[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])
    noisy_img = tf.pad(noisy_img, paddings, mode="SYMMETRIC")
    noisy_img = tf.cast(noisy_img, dtype="float32")

    # Normalise weights
    weights = tf.math.exp(weights)
    weights = tf.divide(weights, (tf.reduce_sum(weights, axis=3, keepdims=True)))

    with tf.Session(""):
        pred = weighted_average.weighted_average(noisy_img, weights).eval()
        return pred


model = tf.keras.models.load_model(sys.argv[1], compile=False)
#feature_list = getFeaturesFromTitle(sys.argv[1])

# Load in patches
patches = tungsten_data.getPatches()

# Extract the model input
test_in = [
    np.array(patches["test"]["noisy"]["diffuse"]),
    np.array(patches["test"]["noisy"]["diffuse_gx"]),
    np.array(patches["test"]["noisy"]["diffuse_gy"]),
    np.array(patches["test"]["noisy"]["diffuse_var"])
]

if config.ALBEDO_DIVIDE:
    test_in = [
        np.array(patches["test"]["noisy"]["albedo_divided"]),
        np.array(patches["test"]["noisy"]["albedo_divided_gx"]),
        np.array(patches["test"]["noisy"]["albedo_divided_gy"]),
        np.array(patches["test"]["noisy"]["albedo_divided_var"])
    ]

feature_list = ["normal", "albedo", "depth"]
for feature in feature_list:
    feature = feature
    # Each feature is split into gradient in X and Y direction, and its
    # corresponding variance
    feature_keys = [feature + "_gx", feature + "_gy", feature + "_var"]
    for key in feature_keys:
        test_in.append(np.array(patches["test"]["noisy"][key]))

model_input = np.concatenate((test_in), 3)
print(model_input.shape)

print("Making prediction... ")
index = random.randint(0, config.NUM_DARTS * config.TEST_SCENES) 
index = 100
weights = model.predict(model_input[index - 1 : index])

noisy_colour = np.array(patches["test"]["noisy"]["diffuse"][index-1 : index])
if config.ALBEDO_DIVIDE:
    noisy_colour = np.array(patches["test"]["noisy"]["albedo_divided"][index-1 : index])

pred = np.zeros(noisy_colour.shape)
preds = applyKernel(np.array(noisy_colour), weights)

# Show the reference, noisy, and denoised image
reference_colour = array_to_img(patches["test"]["reference"]["diffuse"][index - 1 : index][0].clip(0,1))
if config.ALBEDO_DIVIDE:
    reference_colour_albd = array_to_img(patches["test"]["reference"]["albedo_divided"][index - 1 : index][0].clip(0,1))

    albedo = patches["test"]["noisy"]["albedo"][index - 1 : index][0]
    pred = preds[0] * (albedo + 0.00316)

    noisy_colour = noisy_colour[0] * (albedo + 0.00316)

else:
    pred = preds[0]
    noisy_colour = noisy_colour[0]
    print(noisy_colour.shape)


denoised_img = array_to_img(pred.clip(0,1))

print (index)

fig = plt.figure()

fig.add_subplot(1, 3, 1)
plt.imshow(reference_colour)

fig.add_subplot(1, 3, 2)
plt.imshow(array_to_img(noisy_colour.clip(0,1)))

fig.add_subplot(1, 3, 3)
plt.imshow(denoised_img)


plt.show()




