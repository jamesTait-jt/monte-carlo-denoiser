"""Kernel Predicting Convolutional Networks for Denoising Monte Carlo Renderings

Implementation based on this paper:
http://drz.disneyresearch.com/~jnovak/publications/KPCN/KPCN.pdf

This module implements a kernel predicting network to denoise monte carlo
renderings. The aim is to produce high quality renderings at fast speeds by
rendering lowl quality, noisy images from the monte carlo renderer, and then
running the image through the model to remove the noise, producing images at a
much higher quality.
"""

import keras
import math
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.keras.applications.vgg19 import VGG19

import config
from callbacks import TrainValTensorBoard
import data
import weighted_average

class Denoiser():
    """Class for image denoiser via CNN

    This class is designed to read in noisy and reference image data and learn
    the relationship between them with the goal to denoise future, unseen noisy
    images.

    Based on ideas from this paper:
        http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/

    Attributes:

    """
    def __init__(self, train_data, test_data, **kwargs):

        # --- Data hyperparameters --- #

        # The data dictionary
        self.train_data = train_data
        self.test_data = test_data

        # The height and width of the image patches (defaults to 64)
        self.patch_width = kwargs.get("patch_width", 64)
        self.patch_height = kwargs.get("patch_height", 64)

        # Number of input/output channels (defaults to 3 for rgb)
        self.input_channels = kwargs.get("input_channels", 3)
        self.output_channels = kwargs.get("output_channels", 3)

        # Which extra features are to be used by the network
        self.feature_list = kwargs.get("feature_list", [])

        # Prune the data to remove any of the features we don't want
        self.setInputAndLabels()

        # --- Network hyperparameters --- #
        # Padding used in convolutional layers
        self.padding_type = "SAME"

        # General network hyperparameters
        self.curr_batch = 0
        self.batch_size = kwargs.get("batch_size", 5)
        self.num_filters = kwargs.get("num_filters", 100)
        self.kernel_size = kwargs.get("kernel_size", [5, 5])
        self.batch_norm = kwargs.get("batch_norm", False)
        self.num_epochs = kwargs.get("num_epochs", 100)

        # The adam optimiser is used, this block defines its parameters
        self.adam_lr = kwargs.get("adam_lr", 1e-5)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.999)
        self.adam_lr_decay = kwargs.get("adam_lr_decay", 0.0)
        
        self.initialiser_seed = kwargs.get("initialiser_seed", 5678)
        self.kernel_initialiser = tf.keras.initializers.glorot_normal(seed=self.initialiser_seed)
        self.adam = tf.keras.optimizers.Adam(
            lr=self.adam_lr,
            beta_1=self.adam_beta1,
            beta_2=self.adam_beta2,
            decay=self.adam_lr_decay,
            clipnorm=1,
            clipvalue=0.05
            #amsgrad=True
        )

        # Are we using KPCN
        self.kernel_predict = kwargs.get("kernel_predict", False)
        self.kpcn_size = kwargs.get("kpcn_size", 21)

        # Which layer of vgg do we extract features from
        self.vgg_mode = kwargs.get("vgg_mode", 22)

        # Set the log directory with a timestamp
        #self.log_dir = "logs/{}".format(time())
        self.set_log_dir()
        self.set_model_dir()

        # Use the sequential model API
        self.model = kwargs.get("model", tf.keras.models.Sequential())

        # Set callbacks
        self.set_callbacks()

    # Set the directory where logs will be store, using hyperparameters and a
    # timestamp to make them distinct
    def set_log_dir(self):
        log_dir = "logs/"
        for feature in self.feature_list:
            log_dir += (feature + "+")

        log_dir += ("lr:" + str(self.adam_lr) + "+")
        log_dir += ("lr_decay:" + str(self.adam_lr_decay) + "+")
        log_dir += ("bn:" + str(self.batch_norm) + "+")

        self.log_dir = log_dir + "{}".format(time())

    # Set the directory where the models will be stored. (Taken from the
    # hyperparameters)
    def set_model_dir(self):
        model_dir = "models/"
        for feature in self.feature_list:
            model_dir += (feature + "+")

        model_dir += ("lr:" + str(self.adam_lr) + "+")
        model_dir += ("lr_decay:" + str(self.adam_lr_decay) + "+")
        model_dir += ("bn:" + str(self.batch_norm) + "+")

        self.model_dir = model_dir + "{}".format(time())

    def set_callbacks(self):
        self.callbacks = []

        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )

        tensorboard_cb = TrainValTensorBoard(log_dir=self.log_dir, write_graph=True)
        self.callbacks.append(tensorboard_cb)

        filepath = self.model_dir
        model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=1
        )

        self.callbacks.append(model_checkpoint_cb)

        # Stop taining if we don't see an improvement after 20 epochs and
        # restore the best performing weight
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=20,
            restore_best_weights=True
        )
        #self.callbacks.append(early_stopping_cb)

    # Read in the data from the dictionary, exctracting the necessary features
    def setInputAndLabels(self):

        new_train_in = [
            np.array(self.train_data["noisy_colour"]),
            np.array(self.train_data["noisy_colour_gradx"]),
            np.array(self.train_data["noisy_colour_grady"]),
            np.array(self.train_data["noisy_colour_var"])
        ]

        new_test_in = [
            np.array(self.test_data["noisy_colour"]),
            np.array(self.test_data["noisy_colour_gradx"]),
            np.array(self.test_data["noisy_colour_grady"]),
            np.array(self.test_data["noisy_colour_var"])
        ]

        for feature in self.feature_list:
            feature = "noisy_" + feature
            # Each feature is split into gradient in X and Y direction, and its
            # corresponding variance
            feature_keys = [feature + "_gradx", feature + "_grady", feature + "_var"]
            for key in feature_keys:
                new_train_in.append(np.array(self.train_data[key]))
                new_test_in.append(np.array(self.test_data[key]))

        self.train_input = np.concatenate((new_train_in), 3)
        self.test_input = np.concatenate((new_test_in), 3)

        train_labels = [
            np.array(self.train_data["reference_colour"]),
            np.array(self.train_data["noisy_colour"])
        ]

        test_labels = [
            np.array(self.test_data["reference_colour"]),
            np.array(self.test_data["noisy_colour"])
        ]

        #self.train_labels = np.concatenate((train_labels), 3)
        #self.test_labels = np.concatenate((test_labels), 3)

        self.train_labels = np.array(self.train_data["reference_colour"])
        self.test_labels = np.array(self.test_data["reference_colour"])

        # Ensure input channels is the right size
        self.input_channels = self.train_input.shape[3]


    # First convolutional layer (must define input shape)
    def initialConvLayer(self):
        self.model.add(
            tf.keras.layers.Conv2D(
                input_shape=(self.patch_height, self.patch_width, self.input_channels),
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                use_bias=True,
                strides=(1, 1),
                padding=self.padding_type,
                activation="relu",
                kernel_initializer=self.kernel_initialiser # Xavier uniform
            )
        )

    # Convolutional layer (not final)
    def convLayer(self):
        self.model.add(
            tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                use_bias=True,
                strides=[1, 1],
                padding=self.padding_type,
                activation="relu",
                kernel_initializer=self.kernel_initialiser # Xavier uniform
            )
        )
    
    def returnConvLayer(self, prev_layer):
        new_layer = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                use_bias=True,
                strides=[1, 1],
                padding=self.padding_type,
                activation="relu",
                kernel_initializer=self.kernel_initialiser, # Xavier uniform
                #kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(prev_layer)
        
        return new_layer

    def convWithBatchNorm(self):
        # We don't need to add bias if we use batch normalisation
        self.model.add(
            tf.keras.layers.Conv2D(
                input_shape=(self.patch_height, self.patch_height, self.input_channels),
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                use_bias=False,
                strides=(1, 1),
                padding=self.padding_type,
                activation=None,
                kernel_initializer=self.kernel_initialiser # Xavier uniform
            )
        )

        # Batch normalise after the convolutional layer
        self.model.add(tf.keras.layers.BatchNormalization())

        # Apply the relu activation function
        self.model.add(tf.keras.layers.Activation("relu"))


    # Final convolutional layer - no activation function
    def finalConvLayer(self):

        if self.kernel_predict:
            output_size = pow(self.kpcn_size, 2)
        else:
            output_size = self.output_channels

        self.model.add(
            tf.keras.layers.Conv2D(
                filters=output_size,
                kernel_size=self.kernel_size,
                use_bias=True,
                strides=(1, 1),
                padding=self.padding_type,
                activation=None,
                kernel_initializer=self.kernel_initialiser, # Xavier uniform
                #kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )
        )


    def returnFinalConvLayer(self, prev_layer):
        if self.kernel_predict:
            output_size = pow(self.kpcn_size, 2)
        else:
            output_size = self.output_channels

        new_layer = tf.keras.layers.Conv2D(
            filters=output_size,
            kernel_size=self.kernel_size,
            use_bias=True,
            strides=(1, 1),
            padding=self.padding_type,
            activation=None,
            kernel_initializer=self.kernel_initialiser, # Xavier uniform
            #kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(prev_layer)

        return new_layer
        
    def trainBatchNorm(self):
        for _ in range(8):
            self.convWithBatchNorm()
        self.finalConvLayer()

    # Processes the input to the conv network ready for kernel prediction
    def processImgForKernelPrediction(self, noisy_img):
        # Slice the noisy image out of the input
        noisy_img = noisy_img[:, :, :, 0:3]

        # Get the radius of the kernel
        kernel_radius = int(math.floor(self.kpcn_size / 2.0))

        # Pad the image on either side so that the kernel can reach all pixels
        paddings = tf.constant([[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])
        noisy_img = tf.pad(noisy_img, paddings, mode="SYMMETRIC")

        return noisy_img

    # Process the weights ready to be used in kernel prediction
    def processWeightsForKernelPrediction(self, weights):
        # Normalise the weights (softmax)
        exp = tf.math.exp(weights)
        weight_sum = tf.reduce_sum(exp, axis=3, keepdims=True)
        weights = tf.divide(exp, weight_sum)
        return weights


    # Calculates the Peak Signal-to-noise value between two images
    def kernelPredictPSNR(self, noisy_img):
        noisy_img = self.processImgForKernelPrediction(noisy_img)
        def psnr(y_true, y_pred):
            weights = self.processWeightsForKernelPrediction(y_pred)
            prediction = weighted_average.weighted_average(noisy_img, weights)
            return tf.image.psnr(y_true, prediction, max_val=1.0)
        return psnr

    def psnr(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    def MAVKernelPredict(self, noisy_img):
        noisy_img = self.processImgForKernelPrediction(noisy_img)
        def loss(y_true, y_pred):
            weights = self.processWeightsForKernelPrediction(y_pred)
            prediction = weighted_average.weighted_average(noisy_img, weights)
            feature_loss = self.VGG19FeatureLoss(y_true, prediction)
            mse_loss = tf.keras.losses.mean_squared_error(y_true, prediction)
            return feature_loss
        return loss

    # Compares the features from VGG19 of the prediction and ground truth
    def VGG19FeatureLoss(self, y_true, y_pred):
        vgg19 = VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=(config.PATCH_HEIGHT, config.PATCH_WIDTH, 3)
        )
        vgg19.trainable = False
        for layer in vgg19.layers:
            layer.trainable = False

        if self.vgg_mode == 54:
            feature_extractor = tf.keras.Model(
                inputs=vgg19.input,
                outputs=vgg19.get_layer("block5_conv4").output
            )
            feature_shape = [4, 4]
        elif self.vgg_mode == 22:
            feature_extractor = tf.keras.Model(
                inputs=vgg19.input,
                outputs=vgg19.get_layer("block2_conv2").output
            )
            feature_shape = [32, 32]
        feature_extractor.trainable = False

        features_pred = feature_extractor(y_pred)
        features_true = feature_extractor(y_true)

        feature_loss = tf.keras.losses.mean_squared_error(features_pred, features_true)

        # 0.03 rescales to be similar to MSE loss values
        feature_loss = 0.03 * tf.reduce_sum(feature_loss) / (feature_shape[0] * feature_shape[1])

        return feature_loss

    def dropoutLayer(self, rate):
        self.model.add(tf.keras.layers.Dropout(rate))

    def buildNetwork(self):
        if self.batch_norm:
            self.trainBatchNorm()
        else:
            conv_input = tf.keras.layers.Input(
                shape=(self.patch_height, self.patch_width, self.input_channels)
            )
            
            x = self.returnConvLayer(conv_input)
            x = self.returnConvLayer(x)
            x = self.returnConvLayer(x)
            x = self.returnConvLayer(x)
            x = self.returnConvLayer(x)
            x = self.returnConvLayer(x)
            x = self.returnConvLayer(x)
            x = self.returnConvLayer(x)
            pred = self.returnFinalConvLayer(x)

            self.model = tf.keras.models.Model(inputs=conv_input, outputs=pred)

            if self.kernel_predict:
                loss = self.MAVKernelPredict(conv_input)
                metrics = [self.kernelPredictPSNR(conv_input)]
            else:
                loss = "mean_squared_error"
                metrics = [self.psnr]


            self.model.compile(
                optimizer=self.adam,
                loss=loss,
                metrics=metrics
            )

    def train(self):
        self.model.fit(
            self.train_input,
            self.train_labels,
            validation_data=(self.test_input, self.test_labels),
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=self.callbacks
        )

        self.model.save(self.model_dir)

    # Makes a prediction given a noisy image
    def predict(self, test_data):
        model_input = []
        for key in test_data:
            if not key == "reference_colour":
                model_input.append(np.array(test_data[key]))
        model_input = np.concatenate((model_input), 3)
        pred = self.model.predict(model_input)
        return pred

    def eval(self, verbose):
        score = self.model.evaluate(self.test_input, self.test_labels, verbose=0)
        if verbose:
            print(" ")
            print(" ===== DENOISER EVALUATION ===== ")
            print(" ==== Test loss: " + str(score[0]) + " ==== ")
            print(" ==== Test PSNR: " + str(score[1]) + " ==== ")
            print(" ")
        return score

