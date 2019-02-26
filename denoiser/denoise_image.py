import tensorflow as tf
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt

# Contains the training and test data
import data

# Load in the trained model
model = tf.keras.models.load_model("model.h5")

# Load in the noisy and reference validation data to compare
noisy_test = data.data["test"]["colour"]["noisy"]
reference_test = data.data["test"]["colour"]["reference"]

# Make a prediction using the model
pred = model.predict(noisy_test)

# Show the reference, noisy, and denoised image
index = 7 
reference_img = array_to_img(reference_test[index])
noisy_img = array_to_img(noisy_test[index])
denoised_img = array_to_img(pred[index])

fig = plt.figure()

fig.add_subplot(1, 3, 1)
plt.imshow(reference_img)

fig.add_subplot(1, 3, 2)
plt.imshow(noisy_img)

fig.add_subplot(1, 3, 3)
plt.imshow(denoised_img)

plt.show()
