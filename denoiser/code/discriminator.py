import numpy as np
import keras
import tensorflow as tf
from time import time

from callbacks import TrainValTensorBoard
import config

class Discriminator():
    """Class to discriminate between real and fake reference images

    This class is trained to recognize reference (not noisy) images, and
    determine whether an image passed in is a real reference image, or a fake
    one

    """

    def __init__(
        self,
        train_data,
        test_data,
        **kwargs
    ):
        # --- Data hyperparameters --- #

        # Data dictionary
        self.train_data = train_data
        self.test_data = test_data

        #self.setInputAndOutputData()

        # The height and width of the image patches (defaults to 64)
        self.patch_width = kwargs.get("patch_width", config.PATCH_WIDTH)
        self.patch_height = kwargs.get("patch_height", config.PATCH_HEIGHT)

        # Number of input/output channels (defaults to 3 for rgb)
        self.input_channels = kwargs.get("input_channels", 3)

        self.lrelu_activation = kwargs.get("lrelu_activation", 0.2)

        # The adam optimiser is used, this block defines its parameters
        self.adam_lr = kwargs.get("adam_lr", 1e-4)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.999)
        self.adam_lr_decay = kwargs.get("adam_lr_decay", 0.0)

        self.initialiser_seed = kwargs.get("initialiser_seed", 91011)
        self.kernel_initialiser = tf.keras.initializers.glorot_normal() #seed=self.initialiser_seed)

        self.adam = tf.keras.optimizers.Adam(
            lr=self.adam_lr,
            beta_1=self.adam_beta1,
            beta_2=self.adam_beta2,
            decay=self.adam_lr_decay,
            clipnorm=1,
            clipvalue=0.05
        )

        self.num_epochs = kwargs.get("num_epochs", 10)
        self.padding_type = kwargs.get("padding_type", "SAME")
        self.batch_size = kwargs.get("batch_size", 5)

        # Use the sequential model API
        self.model = kwargs.get("model", tf.keras.models.Sequential())

        self.log_dir = "../logs/discriminator/{}".format(time())

        self.setInputAndOutputData()
        self.setCallbacks()

    def setInputAndOutputData(self):

        train_reference_labels = np.ones([len(self.train_data["reference"]["diffuse"]), 1])
        train_fake_labels = np.zeros([len(self.train_data["noisy"]["diffuse"]), 1])

        if config.ALBEDO_DIVIDE:
            self.train_input = np.concatenate(
                (self.train_data["reference"]["albedo_divided"],
                 self.train_data["noisy"]["albedo_divided"])
            )
        else:
            self.train_input = np.concatenate(
                (self.train_data["reference"]["diffuse"],
                 self.train_data["noisy"]["diffuse"])
            )

        self.train_labels = np.concatenate((train_reference_labels, train_fake_labels))

        test_reference_labels = np.ones([len(self.test_data["reference"]["diffuse"]), 1])
        test_fake_labels = np.zeros([len(self.test_data["noisy"]["diffuse"]), 1])

        if config.ALBEDO_DIVIDE:
            self.test_input = np.concatenate(
                (self.test_data["reference"]["albedo_divided"],
                 self.test_data["noisy"]["albedo_divided"])
            )
        else:
            self.test_input = np.concatenate(
                (self.test_data["reference"]["diffuse"],
                 self.test_data["noisy"]["diffuse"])
            )

        self.test_labels = np.concatenate((test_reference_labels, test_fake_labels))

    def setCallbacks(self):
        self.callbacks = []

        tensorboard_cb = TrainValTensorBoard(log_dir=self.log_dir, write_graph=True)
        self.callbacks.append(tensorboard_cb)

        # Stop taining if we don't see an improvement (aabove 98%) after 20 epochs and
        # restore the best performing weight
        #early_stopping_cb = keras.callbacks.EarlyStopping(
        #    monitor='val_acc',
        #    mode='max',
        #    verbose=1,
        #    patience=20,
        #    baseline=0.98,
        #    restore_best_weights=True
        #)
        #self.callbacks.append(early_stopping_cb)

    def initialConvLayer(self, kernel_size, num_filters, strides):
        self.model.add(
            tf.keras.layers.Conv2D(
                input_shape=(self.patch_height, self.patch_width, self.input_channels),
                filters=num_filters,
                kernel_size=kernel_size,
                use_bias=True,
                strides=strides,
                padding=self.padding_type,
                kernel_initializer=self.kernel_initialiser, # Xavier uniform
                #kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )
        )
        self.leakyReLU()

    def leakyReLU(self):
        self.model.add(tf.keras.layers.LeakyReLU(alpha=self.lrelu_activation))

    def dropoutLayer(self, rate):
        self.model.add(tf.keras.layers.Dropout(rate))

    def batchNormalisation(self):
        self.model.add(tf.keras.layers.BatchNormalization())

    def discriminatorBlock(self, kernel_size, num_filters, strides):
        self.model.add(
            tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=kernel_size,
                use_bias=True,
                strides=strides,
                padding=self.padding_type,
                kernel_initializer=self.kernel_initialiser, # Xavier uniform
                #kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )
        )
        self.batchNormalisation()
        self.leakyReLU()

    def denseLayer(self, units):
        self.model.add(tf.keras.layers.Dense(units))

    def sigmoid(self):
        self.model.add(tf.keras.layers.Activation("sigmoid"))

    def flatten(self):
        self.model.add(tf.layers.Flatten())

    def buildNetwork(self):
        self.initialConvLayer(3, 64, [1, 1])

        self.discriminatorBlock(3, 64, [2, 2])
        self.discriminatorBlock(3, 128, [1, 1])
        self.discriminatorBlock(3, 128, [2, 2])
        self.discriminatorBlock(3, 256, [1, 1])
        self.discriminatorBlock(3, 256, [2, 2])
        #self.discriminatorBlock(3, 512, [1, 1])
        #self.discriminatorBlock(3, 512, [2, 2])

        self.denseLayer(1024)
        self.leakyReLU()
        self.flatten()
        self.denseLayer(1)
        self.sigmoid()

    def compile(self):
        self.model.compile(
            optimizer=self.adam,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def train(self):
        print(self.callbacks)
        self.model.fit(
            self.train_input,
            self.train_labels,
            validation_data=(self.test_input, self.test_labels),
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=self.callbacks
        )

    def eval(self):
        score = self.model.evaluate(self.test_input, self.test_labels, verbose=0)
        print(" ")
        print(" ===== DISCRIMINATOR EVALUATION ===== ")
        print(" ====== Test loss: " + str(score[0]) + " ======= ")
        print(" ====== Test acc : " + str(score[1]) + " ======= ")
        print(" ")
