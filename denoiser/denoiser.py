"""Kernel Predicting Convolutional Networks for Denoising Monte Carlo Renderings

Implementation based on this paper:
http://drz.disneyresearch.com/~jnovak/publications/KPCN/KPCN.pdf

This module implements a kernel predicting network to denoise monte carlo
renderings. The aim is to produce high quality renderings at fast speeds by
rendering lowl quality, noisy images from the monte carlo renderer, and then
running the image through the model to remove the noise, producing images at a
much higher quality.
"""

import os
import math
from time import time
import tensorflow as tf
#import data
import make_patches
import config
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19

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
        self.set_input_and_output_data()

        # --- Network hyperparameters --- #
        # Padding used in convolutional layers
        self.padding_type = "SAME"

        # General network hyperparameters
        self.curr_batch = 0
        self.batch_size = kwargs.get("batch_size", 5)
        self.num_epochs = kwargs.get("num_epochs", 100)
        self.num_filters = kwargs.get("num_filters", 100)
        self.kernel_size = kwargs.get("kernel_size", [5, 5])
        self.batch_norm = kwargs.get("batch_norm", False)

        # The adam optimiser is used, this block defines its parameters
        self.adam_lr = kwargs.get("adam_lr", 1e-5)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.999)
        self.adam_lr_decay = kwargs.get("adam_lr_decay", 0.0)
        
        self.adam = tf.keras.optimizers.Adam(
            lr=self.adam_lr,
            beta_1=self.adam_beta1,
            beta_2=self.adam_beta2,
            decay=self.adam_lr_decay
        )

        # Are we using KPCN
        self.kernel_predict = kwargs.get("kernel_predict", False)
        self.kpcn_size = kwargs.get("kpcn_size", 20)

        # Set the log directory with a timestamp
        #self.log_dir = "logs/{}".format(time())
        self.set_log_dir()
        self.set_model_dir()
        
        # Use the sequential model API
        self.model = tf.keras.models.Sequential()

        # Set callbacks
        self.set_callbacks()

    # Set the directory where logs will be store, using hyperparameters and a
    # timestamp to make them distinct
    def set_log_dir(self):
        log_dir = "logs/"
        for feature in self.feature_list:
            log_dir += (feature + "&")

        log_dir += ("lr:" + str(self.adam_lr) + "&")
        log_dir += ("lr_decay:" + str(self.adam_lr_decay) + "&")
        log_dir += ("bn:" + str(self.batch_norm) + "&")

        self.log_dir = log_dir + "{}".format(time())

    # Set the directory where the models will be stored. (Taken from the
    # hyperparameters)
    def set_model_dir(self):
        model_dir = "models/"
        for feature in self.feature_list:
            model_dir += (feature + "&")

        model_dir += ("lr:" + str(self.adam_lr) + "&")
        model_dir += ("lr_decay:" + str(self.adam_lr_decay) + "&")
        model_dir += ("bn:" + str(self.batch_norm) + "&")

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


    # Read in the data from the dictionary, exctracting the necessary features
    def set_input_and_output_data(self):

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
        
        self.train_output = np.array(self.train_data["reference_colour"])
        self.test_output = np.array(self.test_data["reference_colour"])

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
                kernel_initializer="glorot_uniform" # Xavier uniform
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
                kernel_initializer="glorot_uniform" # Xavier uniform
            )
        )

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
                kernel_initializer="glorot_uniform" # Xavier uniform
            )
        )

        # Batch normalise after the convolutional layer
        self.model.add(
            tf.keras.layers.BatchNormalization()    
        )

        # Apply the relu activation function
        self.model.add(
            tf.keras.layers.Activation("relu")
        )


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
                kernel_initializer="glorot_uniform" # Xavier uniform
            )
        )

        if self.kernel_predict:
            self.model.add(
                tf.keras.layers.Lambda(self.kernelPrediction)
            )

    def trainBatchNorm(self):
        for i in range(8):
            self.convWithBatchNorm()
        self.finalConvLayer()

    def kernelPrediction(self, x):
        exp = tf.math.exp(x)
        weight_sum = tf.math.reduce_sum(exp, axis=3, keepdims=True)
        weight_avg = tf.math.divide(exp, weight_sum)
 
        #kernel_radius = int(math.floor(self.kpcn_size / 2.0))

        #input_img = tf.slice(self.model.input[self.curr_batch : self.batch_size], [0, 0, 0], [config.PATCH_HEIGHT, config.PATCH_WIDTH, 3])

        # Symmetric padding means border pixels will just have their inside
        # neighbours mirrored onto the outside
        #paddings = tf.constant([[kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0,0]])
        #input_img = tf.pad(input_img, paddings, mode="SYMMETRIC")

        return weight_avg

    # Calculates the Peak Signal-to-noise value between two images
    def psnr(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    def perceptualLoss(self, y_true, y_pred):
        vgg19 = VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg19.trainable = False
        for layer in vgg19.layers:
            layer.trainable = False

        # create a model that ouputs the features from level 'block2_conv2'
        feature_extractor = tf.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer("block2_conv2").output)

        features_pred = feature_extractor(y_pred)
        features_true = feature_extractor(y_true)

        return 0.006 * tf.math.reduce_mean(tf.square(features_pred - features_true), axis=-1)

    def train(self):

        if self.batch_norm:
            self.trainBatchNorm()

        else:
            self.initialConvLayer()
            for _ in range(7):
                self.convLayer()
            self.finalConvLayer()

        self.model.compile(
            optimizer=self.adam,
            #loss="mean_absolute_error",
            loss=self.perceptualLoss,
            metrics=[self.psnr]
        )

        self.model.fit(
            self.train_input,
            self.train_output,
            validation_data=(self.test_input, self.test_output),
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=self.callbacks
        )
    
        self.model.save(self.model_dir)

    def eval(self):
        score = self.model.evaluate(self.test_input, self.test_output, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

# Callback to plot both training and validation metrics
#   https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', 'epoch_'): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def denoise():
    #train_data = data.data["train"]
    #test_data = data.data["test"]

    patches = make_patches.makePatches()
    train_data = patches["train"]
    test_data = patches["test"]

    feature_list = []
    denoiser = Denoiser(
        train_data, 
        test_data, 
        num_epochs=200,
        feature_list=feature_list
    )

    #denoiser.train()
    #denoiser.eval()
    del denoiser

    feature_list = ["sn", "albedo", "depth"]
    denoiser = Denoiser(
        train_data, 
        test_data, 
        num_epochs=500,
        feature_list=feature_list
    )
    denoiser.train()
    denoiser.eval()
    del denoiser

    denoiser = Denoiser(
        train_data,
        test_data,
        num_epochs=50,
        feature_list=feature_list
    )

    #denoiser.train()
    #denoiser.eval()

    del denoiser
