import tensorflow as tf
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np

# Contains the training and test data
import data

# Load in the trained model
model = tf.keras.models.load_model("models/model.h5")

model_input = np.concatenate(
    (
        data.data["test"]["colour"]["noisy"],
        data.data["test"]["colour_gradx"]["noisy"],
        data.data["test"]["colour_grady"]["noisy"],
        data.data["test"]["sn"]["noisy"],
        data.data["test"]["sn_gradx"]["noisy"],
        data.data["test"]["sn_grady"]["noisy"],
        data.data["test"]["albedo"]["noisy"],
        data.data["test"]["albedo_gradx"]["noisy"],
        data.data["test"]["albedo_grady"]["noisy"],
        data.data["test"]["colour_var"]["noisy"],
        data.data["test"]["sn_var"]["noisy"],
        data.data["test"]["albedo_var"]["noisy"]
    ), 3)

# Make a prediction using the model
pred = data.convert_channels_7_to_3(model.predict(model_input))

# Show the reference, noisy, and denoised image
index = 7 
reference_colour = array_to_img(data.data["test"]["colour"]["reference"][index])
noisy_colour = array_to_img(data.data["test"]["colour"]["noisy"][index])

denoised_img = array_to_img(pred[index])

fig = plt.figure()

fig.add_subplot(1, 3, 1)
plt.imshow(reference_colour)

fig.add_subplot(1, 3, 2)
plt.imshow(noisy_colour)

fig.add_subplot(1, 3, 3)
plt.imshow(denoised_img)

plt.show()
