import tensorflow as tf
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np

# Contains the training and test data
import data

# Load in the trained model
model = tf.keras.models.load_model("models/model.h5")

# Load in the noisy and reference validation data to compare
noisy_colour_test = data.data["test"]["colour"]["noisy"]
reference_colour_test = data.data["test"]["colour"]["reference"]

noisy_sn_test = data.data["test"]["surface_normal"]["noisy"]
reference_sn_test = data.data["test"]["surface_normal"]["reference"]

model_input = np.concatenate((noisy_colour_test, noisy_sn_test), 3)

# Make a prediction using the model
pred = model.predict(model_input)

# Show the reference, noisy, and denoised image
index = 23 
reference_colour = array_to_img(reference_colour_test[index])
noisy_colour = array_to_img(noisy_colour_test[index])

denoised_img = array_to_img(pred[index])

fig = plt.figure()

fig.add_subplot(1, 3, 1)
plt.imshow(reference_colour)

fig.add_subplot(1, 3, 2)
plt.imshow(noisy_colour)

fig.add_subplot(1, 3, 3)
plt.imshow(denoised_img)

plt.show()
