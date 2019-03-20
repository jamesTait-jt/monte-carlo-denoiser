import tensorflow as tf
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np
import sys
import random

import make_patches
import denoise_full_image
import config

# Load in the trained model
model = tf.keras.models.load_model(sys.argv[1], compile=False)
feature_list = denoise_full_image.getFeaturesFromTitle(sys.argv[1])

patches = make_patches.makePatches()

test_in = [
    np.array(patches["test"]["noisy_colour"]),
    np.array(patches["test"]["noisy_colour_gradx"]),
    np.array(patches["test"]["noisy_colour_grady"]),
    np.array(patches["test"]["noisy_colour_var"])
]

for feature in feature_list:
    feature_keys = [feature + "_gradx", feature + "_grady", feature + "_var"]
    for key in feature_keys:
        if key.endswith("var") or "depth" in key.split('_'):
            test_in.append(patches["test"]["noisy_" + key])
        else:
            test_in.append(patches["test"]["noisy_" + key])

model_input = np.concatenate((test_in), 3)
pred = model.predict(model_input)

# Show the reference, noisy, and denoised image
index = random.randint(config.NUM_DARTS) 
reference_colour = array_to_img(patches["test"]["reference_colour"][index])
noisy_colour = array_to_img(patches["test"]["noisy_colour"][index])
denoised_img = array_to_img(pred[index])

fig = plt.figure()

fig.add_subplot(1, 3, 1)
plt.imshow(reference_colour)

fig.add_subplot(1, 3, 2)
plt.imshow(noisy_colour)

fig.add_subplot(1, 3, 3)
plt.imshow(denoised_img)

plt.show()

print("Index was: " + str(index))
