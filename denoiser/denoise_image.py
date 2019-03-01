import tensorflow as tf
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np

# Contains the training and test data
import data

# Load in the trained model
model = tf.keras.models.load_model("models/model.h5")

# Load in the noisy and reference validation data to compare
# Colour information (RGB picture)
reference_colour_test = data.data["test"]["colour"]["reference"]
noisy_colour_test = data.data["test"]["colour"]["noisy"]

reference_colour_gradx_test = data.data["test"]["colour_gradx"]["reference"]
noisy_colour_gradx_test = data.data["test"]["colour_gradx"]["noisy"]

reference_colour_grady_test = data.data["test"]["colour_grady"]["reference"]
noisy_colour_grady_test = data.data["test"]["colour_grady"]["noisy"]

# Colour variance (3 channels converted to 1 by calculating luminance) 
reference_colour_var_test = data.data["test"]["colour_var"]["reference"]
noisy_colour_var_test = data.data["test"]["colour_var"]["noisy"]

# Surface normals
reference_sn_test = data.data["test"]["surface_normal"]["reference"]
noisy_sn_test = data.data["test"]["surface_normal"]["noisy"]

reference_sn_gradx_test = data.data["test"]["surface_normal_gradx"]["reference"]
noisy_sn_gradx_test = data.data["test"]["surface_normal_gradx"]["noisy"]

reference_sn_grady_test = data.data["test"]["surface_normal_grady"]["reference"]
noisy_sn_grady_test = data.data["test"]["surface_normal_grady"]["noisy"]

model_input = np.concatenate(
    (
        noisy_colour_test, 
        noisy_colour_gradx_test,
        noisy_colour_grady_test,
        noisy_colour_var_test,
        noisy_sn_test,
        noisy_sn_gradx_test,
        noisy_sn_grady_test
    ), 3)

# Make a prediction using the model
pred = data.convert_channels_7_to_3(model.predict(model_input))

# Show the reference, noisy, and denoised image
index = 7 
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
