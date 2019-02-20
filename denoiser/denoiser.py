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

mnist = tf.keras.datasets.mnist

# Global flags
FLAGS = tf.app.flags

tf.app.flags.DEFINE_integer ("patchSize", 65,
                            "The size of the input patches")

tf.app.flags.DEFINE_integer ("reconstructionKernelSize", 21,
                            "The size of the reconstruction kernel")

tf.app.flags.DEFINE_integer ("inputChannels", 27,
                            "The number of channels in an input patch")

tf.app.flags.DEFINE_integer ("outputChannels", 3,
                            "The number of channels in an output patch")

tf.app.flags.DEFINE_float   ("learningRate", 0.00001,
                            "The learning rate for ADAM")

tf.app.flags.DEFINE_integer ("numEpochs", 200,
                            "Number of training epochs")

# Convolutional layer (not final)
def convLayer():
    return tf.keras.layers.Conv2D(
        filters=100,
        kernel_size=(5, 5),
        use_bias=True,
        strides=[1, 1, 1, 1],
        padding="VALID",
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.glorot_uniform
    )

def finalConvLayer():
    return tf.keras.layers.Conv2D(
        filters=hp.final_layer_size,
        kernel_size=None,
        use_bias=True,
        strides=[1, 1, 1, 1],
        padding="VALID",
        activation=None,
        kernel_initializer=tf.keras.initializers.glorot_uniform
    )

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

my_model = tf.keras.models.Sequential([

    # Conv layer 1
    convLayer(),

    # Conv layer 2
    convLayer(),

    # Conv layer 3
    convLayer(),

    # Conv layer 4
    convLayer(),

    # Conv layer 5
    convLayer(),

    # Conv layer 6
    convLayer(),

    # Conv layer 7
    convLayer(),

    # Conv layer 8
    finalConvLayer()
])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

