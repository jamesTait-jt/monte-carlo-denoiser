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
import keras.backend as K
import math
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.python import debug as tf_debug

import config
import data
import models
import metrics_module
import train_util
import weighted_average

from callbacks import TrainValTensorBoard


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
        self.patch_width = kwargs.get("patch_width", config.PATCH_WIDTH)
        self.patch_height = kwargs.get("patch_height", config.PATCH_HEIGHT)

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
        self.num_layers = kwargs.get("num_layers", 8)
        self.bn = kwargs.get("bn", False)

        # The adam optimiser is used, this block defines its parameters
        self.adam_lr = kwargs.get("adam_lr", 1e-4)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.999)
        self.adam_lr_decay = kwargs.get("adam_lr_decay", 0.0)
        
        self.activation = keras.layers.ReLU()
        self.loss = kwargs.get("loss", "mae")
        self.early_stopping = kwargs.get("early_stopping", False)

        self.initialiser_seed = kwargs.get("initialiser_seed", 5678)
        self.kernel_initialiser = keras.initializers.glorot_normal(seed=self.initialiser_seed)
        self.adam = keras.optimizers.Adam(
            lr=self.adam_lr,
            beta_1=self.adam_beta1,
            beta_2=self.adam_beta2,
            decay=self.adam_lr_decay,
            clipvalue=1
            #amsgrad=True
        )

        # Are we using KPCN
        self.kernel_predict = kwargs.get("kernel_predict", False)
        self.kpcn_size = kwargs.get("kpcn_size", 21)

        self.vgg = models.buildVGG()

        # Which layer of vgg do we extract features from
        #self.vgg_mode = kwargs.get("vgg_mode", 22)

        # Set the log directory with a timestamp
        #self.log_dir = "logs/{}".format(time())
        #self.set_log_dir()
        #self.set_model_dir()

        now = time()
        self.model_dir = kwargs.get("model_dir", "default") + str(now)
        self.log_dir = kwargs.get("log_dir", "default") + str(now)

        # Use the sequential model API
        self.model = kwargs.get("model", keras.models.Sequential())

        # Set callbacks
        self.set_callbacks()

    # Set the directory where logs will be store, using hyperparameters and a
    # timestamp to make them distinct
    def set_log_dir(self):
        log_dir = "../logs/"
        for feature in self.feature_list:
            log_dir += (feature + "+")

        log_dir += ("lr:" + str(self.adam_lr) + "+")
        log_dir += ("lr_decay:" + str(self.adam_lr_decay) + "+")
        log_dir += ("bn:" + str(self.batch_norm) + "+")

        self.log_dir = log_dir + "{}".format(time())

    # Set the directory where the models will be stored. (Taken from the
    # hyperparameters)
    def set_model_dir(self):
        model_dir = "../models/"
        for feature in self.feature_list:
            model_dir += (feature + "+")

        model_dir += ("lr:" + str(self.adam_lr) + "+")
        model_dir += ("lr_decay:" + str(self.adam_lr_decay) + "+")
        model_dir += ("bn:" + str(self.batch_norm) + "+")

        self.model_dir = model_dir + "{}".format(time())

    def set_callbacks(self):
        self.callbacks = []

        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )

        tensorboard_cb = TrainValTensorBoard(log_dir=self.log_dir, write_graph=True)
        self.callbacks.append(tensorboard_cb)

        filepath = self.model_dir
        model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=1
        )

        self.callbacks.append(model_checkpoint_cb)

        reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1, 
            patience=10, 
            verbose=1, 
            mode='auto', 
            min_lr=1e-5
        )
        self.callbacks.append(reduce_lr_cb)

        # Stop taining if we don't see an improvement after 20 epochs and
        # restore the best performing weight
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=20,
            restore_best_weights=True
        )
        if self.early_stopping:
            self.callbacks.append(early_stopping_cb)

    # Read in the data from the dictionary, exctracting the necessary features
    def setInputAndLabels(self):

        diffuse_or_albedo_div = "diffuse"
        if config.ALBEDO_DIVIDE:
            diffuse_or_albedo_div = "albedo_divided"

        new_train_in = [
            np.array(self.train_data["noisy"][diffuse_or_albedo_div]),
            np.array(self.train_data["noisy"][diffuse_or_albedo_div + "_gx"]),
            np.array(self.train_data["noisy"][diffuse_or_albedo_div + "_gy"]),
            np.array(self.train_data["noisy"][diffuse_or_albedo_div + "_var"])
        ]

        new_test_in = [
            np.array(self.test_data["noisy"][diffuse_or_albedo_div]),
            np.array(self.test_data["noisy"][diffuse_or_albedo_div + "_gx"]),
            np.array(self.test_data["noisy"][diffuse_or_albedo_div + "_gy"]),
            np.array(self.test_data["noisy"][diffuse_or_albedo_div + "_var"])
        ]

        for feature in self.feature_list:
            feature = feature
            # Each feature is split into gradient in X and Y direction, and its
            # corresponding variance
            feature_keys = [feature + "_gx", feature + "_gy", feature + "_var"]
            for key in feature_keys:
                new_train_in.append(np.array(self.train_data["noisy"][key]))
                new_test_in.append(np.array(self.test_data["noisy"][key]))

        self.train_input = np.concatenate((new_train_in), 3)
        self.test_input = np.concatenate((new_test_in), 3)

        self.train_labels = np.array(self.train_data["reference"][diffuse_or_albedo_div])
        self.test_labels = np.array(self.test_data["reference"][diffuse_or_albedo_div])

        # Ensure input channels is the right size
        self.input_channels = self.train_input.shape[3]


    def psnr(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    def dropoutLayer(self, rate):
        self.model.add(keras.layers.Dropout(rate))

    def kpcnPSNR(self, noisy_img):
        """Apply the kernel and calculate the psnr value between it and the ground
        truth. If we have albedo divide on, this is calculated between the albedo
        divided reference image and the albedo divided prediction. If not, it is
        calculated between normal images."""
        def psnr(y_true, y_pred):
            prediction = train_util.applyKernel(noisy_img, y_pred, self.kpcn_size)

            # Normalise the images to ensure we don't get NaN
            y_true = y_true / K.max(y_true)
            prediction = prediction / K.max(prediction)

            psnr_val = tf.image.psnr(y_true, prediction, max_val=1.0)
            psnr_val = train_util.meanWithoutNanOrInf(psnr_val)
            
            return psnr_val
        return psnr


    def buildNetwork(self):

        def convLayer(prev_layer, num_filters):
            new_layer = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=self.kernel_size,
                use_bias=True,
                strides=[1, 1],
                padding=self.padding_type,
                kernel_initializer=self.kernel_initialiser, # Xavier uniform
            )(prev_layer)
            
            
            return new_layer

        ###########################################################

        noisy_img = keras.layers.Input(shape=config.DENOISER_INPUT_SHAPE)
        
        x = convLayer(noisy_img, self.num_filters)
        x = self.activation(x)
        for _ in range(self.num_layers):
            x = convLayer(x, self.num_filters)
            x = self.activation(x)
            #x = keras.layers.Dropout(0.2)(x)
            if self.bn:
                x = keras.layers.BatchNormalization(momentum=0.8)(x)

        if self.kernel_predict:
            pred = convLayer(x, pow(self.kpcn_size, 2))
        else:
            pred = convLayer(x, 3)

        self.model = keras.models.Model(inputs=noisy_img, outputs=pred)

        if self.kernel_predict:
            # Use the psnr metric
            metrics=[self.kpcnPSNR(noisy_img)]

            # Mean absolute error
            if self.loss == "mae":
                loss = metrics_module.kpcnMAE(noisy_img, self.kpcn_size)
            elif self.loss == "mse":
                loss = metrics_module.kpcnMSE(noisy_img, self.kpcn_size)
            # Feature loss with block2conv2
            elif self.loss == "vgg22":
                loss = metrics_module.kpcnVGG(noisy_img, self.kpcn_size, self.vgg)
            elif self.loss == "ssim":
                loss = metrics_module.kpcnSSIM(noisy_img, self.kpcn_size)
        else:
            loss = "mean_absolute_error"
            #metrics = [self.psnr]

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


