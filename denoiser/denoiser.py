"""Kernel Predicting Convolutional Networks for Denoising Monte Carlo Renderings

Implementation based on this paper:
http://drz.disneyresearch.com/~jnovak/publications/KPCN/KPCN.pdf

This module implements a kernel predicting network to denoise monte carlo
renderings. The aim is to produce high quality renderings at fast speeds by
rendering lowl quality, noisy images from the monte carlo renderer, and then
running the image through the model to remove the noise, producing images at a
much higher quality.
"""

import tensorflow as tf
from keras.preprocessing.image import array_to_img
import data
import numpy as np


########################
##### Global flags #####
########################
FLAGS = tf.app.flags.FLAGS

##### Network #####
tf.app.flags.DEFINE_integer ("patchSize", 64,
                            "The size of the input patches")

#tf.app.flags.DEFINE_integer ("reconstructionKernelSize", 21,
#                            "The size of the reconstruction kernel")

tf.app.flags.DEFINE_integer ("inputChannels", 27,
                            "The number of channels in an input patch")

tf.app.flags.DEFINE_integer ("outputChannels", 27,
                            "The number of channels in an output patch")

tf.app.flags.DEFINE_float   ("learningRate", 0.00001,
                            "The learning rate for ADAM")

tf.app.flags.DEFINE_integer ("batchSize", 5,
                            "Number of patches per minibatch")

tf.app.flags.DEFINE_integer ("numEpochs", 500,
                            "Number of training epochs")

tf.app.flags.DEFINE_integer ("numFilters", 100,
                            "Number of filters in the hidden layers")

tf.app.flags.DEFINE_integer ("kernelSize", 5,
                            "Width and height of the convolution kernels")

##### Filesystem #####
tf.app.flags.DEFINE_string  ("modelSaveDir", "models",
                            "Location at which the models are stored")

# First convolutional layer (must define input shape)
def firstConvLayer(model):
    model.add(
        tf.keras.layers.Conv2D(
            input_shape=(FLAGS.patchSize, FLAGS.patchSize, FLAGS.inputChannels),
            filters=FLAGS.numFilters,
            kernel_size=FLAGS.kernelSize,
            use_bias=True,
            strides=(1, 1),
            padding="SAME",
            activation="relu",
            kernel_initializer="glorot_uniform" # Xavier uniform
        )
    )

# Convolutional layer (not final)
def convLayer(model):
    model.add(
        tf.keras.layers.Conv2D(
            filters=FLAGS.numFilters,
            kernel_size=FLAGS.kernelSize,
            use_bias=True,
            strides=[1, 1],
            padding="SAME",
            activation="relu",
            kernel_initializer="glorot_uniform" # Xavier uniform
        )
    )

def convWithBatchNorm(model):
    # We don't need to add bias if we use batch normalisation
    model.add(
        tf.keras.layers.Conv2D(
            input_shape=(FLAGS.patchSize, FLAGS.patchSize, FLAGS.inputChannels),
            filters=FLAGS.numFilters,
            kernel_size=FLAGS.kernelSize,
            use_bias=False,
            strides=(1, 1),
            padding="SAME",
            activation=None,
            kernel_initializer="glorot_uniform" # Xavier uniform
        )
    )

    # Batch normalise after the convolutional layer
    model.add(
        tf.keras.layers.BatchNormalization()    
    )

    # Apply the relu activation function
    model.add(
        tf.keras.layers.Activation("relu")
    )


# Final convolutional layer - no activation function
def finalConvLayer(model):
    model.add(
        tf.keras.layers.Conv2D(
            filters=FLAGS.outputChannels,
            kernel_size=FLAGS.kernelSize,
            use_bias=True,
            strides=(1, 1),
            padding="SAME",
            activation=None,
            kernel_initializer="glorot_uniform" # Xavier uniform
        )
    )


model = tf.keras.models.Sequential()

# Conv layer 1
firstConvLayer(model)

# Conv layer 2
convLayer(model)
#convWithBatchNorm(model)

# Conv layer 3
convLayer(model)
#convWithBatchNorm(model)

# Conv layer 4
convLayer(model)
#convWithBatchNorm(model)

# Conv layer 5
convLayer(model)
#convWithBatchNorm(model)

# Conv layer 6
convLayer(model)
#convWithBatchNorm(model)

# Conv layer 7
convLayer(model)
#convWithBatchNorm(model)

# Conv layer 8
convLayer(model)
#convWithBatchNorm(model)
    
# Conv layer 9
finalConvLayer(model)


tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', 
    histogram_freq=0,  
    write_graph=True, 
    write_images=True
)

print(data.data["train"]["colour"]["noisy"].shape)
print(data.data["train"]["colour_gradx"]["noisy"].shape)
print(data.data["train"]["colour_grady"]["noisy"].shape)
print(data.data["train"]["sn"]["noisy"].shape)
print(data.data["train"]["sn_gradx"]["noisy"].shape)
print(data.data["train"]["sn_grady"]["noisy"].shape)
print(data.data["train"]["albedo"]["noisy"].shape)
print(data.data["train"]["albedo_gradx"]["noisy"].shape)
print(data.data["train"]["albedo_grady"]["noisy"].shape)
print(data.data["train"]["colour_var"]["noisy"].shape)
print(data.data["train"]["sn_var"]["noisy"].shape)
print(data.data["train"]["albedo_var"]["noisy"].shape)

model_input = np.concatenate(
    (
        data.data["train"]["colour"]["noisy"],
        data.data["train"]["colour_gradx"]["noisy"],
        data.data["train"]["colour_grady"]["noisy"],
        #data.data["train"]["sn"]["noisy"],
        data.data["train"]["sn_gradx"]["noisy"],
        data.data["train"]["sn_grady"]["noisy"],
        #data.data["train"]["albedo"]["noisy"],
        data.data["train"]["albedo_gradx"]["noisy"],
        data.data["train"]["albedo_grady"]["noisy"],
        #data.data["train"]["depth"]["noisy"],
        data.data["train"]["depth_gradx"]["noisy"],
        data.data["train"]["depth_grady"]["noisy"],
        data.data["train"]["colour_var"]["noisy"],
        data.data["train"]["sn_var"]["noisy"],
        data.data["train"]["albedo_var"]["noisy"],
        data.data["train"]["depth_var"]["noisy"]
    ), 3)

model_output = np.concatenate(
    (
        data.data["train"]["colour"]["reference"],
        data.data["train"]["colour_gradx"]["reference"],
        data.data["train"]["colour_grady"]["reference"],
        #data.data["train"]["sn"]["reference"],
        data.data["train"]["sn_gradx"]["reference"],
        data.data["train"]["sn_grady"]["reference"],
        #data.data["train"]["albedo"]["reference"],
        data.data["train"]["albedo_gradx"]["reference"],
        data.data["train"]["albedo_grady"]["reference"],
        #data.data["train"]["depth"]["reference"],
        data.data["train"]["depth_gradx"]["reference"],
        data.data["train"]["depth_grady"]["reference"],
        data.data["train"]["colour_var"]["reference"],
        data.data["train"]["sn_var"]["reference"],
        data.data["train"]["albedo_var"]["reference"],
        data.data["train"]["depth_var"]["reference"]
    ), 3)

adam = tf.keras.optimizers.Adam(FLAGS.learningRate)
model.compile(
    optimizer=adam,
    loss="mean_absolute_error",
    metrics=["accuracy"]
)

model.fit(
    model_input,
    model_output,
    batch_size=FLAGS.batchSize,
    epochs=FLAGS.numEpochs,
    callbacks=[tensorboard_callback]
)

model.save(FLAGS.modelSaveDir + "/model.h5")

test_input = np.concatenate(
    (
        data.data["test"]["colour"]["noisy"],
        data.data["test"]["colour_gradx"]["noisy"],
        data.data["test"]["colour_grady"]["noisy"],
        #data.data["test"]["sn"]["noisy"],
        data.data["test"]["sn_gradx"]["noisy"],
        data.data["test"]["sn_grady"]["noisy"],
        #data.data["test"]["albedo"]["noisy"],
        data.data["test"]["albedo_gradx"]["noisy"],
        data.data["test"]["albedo_grady"]["noisy"],
        #data.data["test"]["depth"]["noisy"],
        data.data["test"]["depth_gradx"]["noisy"],
        data.data["test"]["depth_grady"]["noisy"],
        data.data["test"]["colour_var"]["noisy"],
        data.data["test"]["sn_var"]["noisy"],
        data.data["test"]["albedo_var"]["noisy"],
        data.data["test"]["depth_var"]["noisy"]
    ), 3)

test_output = np.concatenate(
    (
        data.data["test"]["colour"]["reference"],
        data.data["test"]["colour_gradx"]["reference"],
        data.data["test"]["colour_grady"]["reference"],
        #data.data["test"]["sn"]["reference"],
        data.data["test"]["sn_gradx"]["reference"],
        data.data["test"]["sn_grady"]["reference"],
        #data.data["test"]["albedo"]["reference"],
        data.data["test"]["albedo_gradx"]["reference"],
        data.data["test"]["albedo_grady"]["reference"],
        #data.data["test"]["depth"]["reference"],
        data.data["test"]["depth_gradx"]["reference"],
        data.data["test"]["depth_grady"]["reference"],
        data.data["test"]["colour_var"]["reference"],
        data.data["test"]["sn_var"]["reference"],
        data.data["test"]["albedo_var"]["reference"],
        data.data["test"]["depth_var"]["reference"]
    ), 3)

score = model.evaluate(test_input, test_output, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

del model
