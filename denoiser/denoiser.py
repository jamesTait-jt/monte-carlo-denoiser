"""Kernel Predicting Convolutional Networks for Denoising Monte Carlo Renderings

Implementation based on this paper:
http://drz.disneyresearch.com/~jnovak/publications/KPCN/KPCN.pdf

This module implements a kernel predicting network to denoise monte carlo
renderings. The aim is to produce high quality renderings at fast speeds by
rendering lowl quality, noisy images from the monte carlo renderer, and then
running the image through the model to remove the noise, producing images at a
much higher quality.
"""

import math
import os
from time import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
import keras
from tqdm import tqdm

import data
import config
#import weighted_average

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
        self.test_pred = test_data

        #self.setInputAndOutputData()

        # The height and width of the image patches (defaults to 64)
        self.patch_width = kwargs.get("patch_width", 64)
        self.patch_height = kwargs.get("patch_height", 64)

        # Number of input/output channels (defaults to 3 for rgb)
        self.input_channels = kwargs.get("input_channels", 3)

        self.lrelu_activation = kwargs.get("lrelu_activation", 0.2)

        # The adam optimiser is used, this block defines its parameters
        self.adam_lr = kwargs.get("adam_lr", 1e-4)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.999)
        self.adam_lr_decay = kwargs.get("adam_lr_decay", 0.0)

        self.adam = tf.keras.optimizers.Adam(
            lr=self.adam_lr,
            beta_1=self.adam_beta1,
            beta_2=self.adam_beta2,
            decay=self.adam_lr_decay
        )

        self.num_epochs = kwargs.get("num_epochs", 10)
        self.padding_type = kwargs.get("padding_type", "SAME")
        self.batch_size = kwargs.get("batch_size", 5)

        # Use the sequential model API
        self.model = kwargs.get("model", tf.keras.models.Sequential())

        self.log_dir = "logs/discriminator/{}".format(time())

        self.setCallbacks()

    def setInputAndOutputData(self):
        train_reference_labels = np.ones([len(self.train_ref_images), 1])
        train_fake_labels = np.zeros([len(self.train_pred), 1])

        self.train_input = np.concatenate((self.train_ref_images, self.train_pred))
        self.train_labels = np.concatenate((train_reference_labels, train_fake_labels))

        test_reference_labels = np.ones([len(self.test_ref_images), 1])
        test_fake_labels = np.zeros([len(self.test_pred), 1])

        self.test_input = np.concatenate((self.test_ref_images, self.test_pred))
        self.test_labels = np.concatenate((test_reference_labels, test_fake_labels))

    def setCallbacks(self):
        self.callbacks = []

        tensorboard_cb = TrainValTensorBoard(log_dir=self.log_dir, write_graph=True)
        self.callbacks.append(tensorboard_cb)

        # Stop taining if we don't see an improvement (aabove 98%) after 20 epochs and
        # restore the best performing weight
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor='val_acc',
            mode='max',
            verbose=1,
            patience=20,
            baseline=0.98,
            restore_best_weights=True
        )
        self.callbacks.append(early_stopping_cb)

    def initialConvLayer(self, kernel_size, num_filters, strides):
        self.model.add(
            tf.keras.layers.Conv2D(
                input_shape=(self.patch_height, self.patch_width, self.input_channels),
                filters=num_filters,
                kernel_size=kernel_size,
                use_bias=True,
                strides=strides,
                padding=self.padding_type,
                kernel_initializer="glorot_uniform" # Xavier uniform
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
                kernel_initializer="glorot_uniform" # Xavier uniform
            )
        )
        self.batchNormalisation()
        self.leakyReLU()
        self.dropoutLayer(0.6)

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
        self.discriminatorBlock(3, 512, [1, 1])
        self.discriminatorBlock(3, 512, [2, 2])

        self.denseLayer(1024)
        self.leakyReLU()
        self.dropoutLayer(0.6)
        self.flatten()
        self.denseLayer(1)
        self.sigmoid()

        self.model.compile(
            optimizer=self.adam,
            loss="binary_crossentropy",
            metrics=["accuracy"]
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

    def eval(self):
        score = self.model.evaluate(self.test_input, self.test_labels, verbose=0)
        print(" ")
        print(" ===== DISCRIMINATOR EVALUATION ===== ")
        print(" ====== Test loss: " + str(score[0]) + " ======= ")
        print(" ====== Test acc : " + str(score[1]) + " ======= ")
        print(" ")


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
        self.vgg_epochs = kwargs.get("vgg_epochs", 100)
        self.mse_epochs = kwargs.get("mse_epochs", 100)
        self.num_filters = kwargs.get("num_filters", 100)
        self.kernel_size = kwargs.get("kernel_size", [5, 5])
        self.batch_norm = kwargs.get("batch_norm", False)

        # Discriminator to help with loss function
        if self.vgg_epochs > 0:
            self.discriminator = kwargs.get("discriminator")

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

        # Stop taining if we don't see an improvement after 20 epochs and
        # restore the best performing weight
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=20,
            restore_best_weights=True
        )
        self.callbacks.append(early_stopping_cb)

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
    
    def returnConvLayer(self, prev_layer):
        new_layer = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                use_bias=True,
                strides=[1, 1],
                padding=self.padding_type,
                activation="relu",
                kernel_initializer="glorot_uniform" # Xavier uniform
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
                kernel_initializer="glorot_uniform" # Xavier uniform
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
                kernel_initializer="glorot_uniform" # Xavier uniform
            )
        )

        #if self.kernel_predict:
        #    self.model.add(
        #        tf.keras.layers.Lambda(self.kernelPrediction)
        #    )

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
            kernel_initializer="glorot_uniform" # Xavier uniform
        )(prev_layer)

        return new_layer
        

    def trainBatchNorm(self):
        for _ in range(8):
            self.convWithBatchNorm()
        self.finalConvLayer()

    def kernelPrediction(self, x, noisy_img):
        exp = tf.math.exp(x)
        weight_sum = tf.reduce_sum(exp, axis=3, keepdims=True)
        weight_avg = tf.divide(exp, weight_sum)
        weights = tf.reshape(weight_avg, shape=[self.batch_size, config.PATCH_HEIGHT, config.PATCH_WIDTH, self.kpcn_size, self.kpcn_size])

        # Slice the noisy image out of the input
        noisy_img = noisy_img[:, :, :, 0:3]

        kernel_radius = int(math.floor(self.kpcn_size / 2.0))
        paddings = tf.constant([[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])
        noisy_img = tf.pad(noisy_img, paddings, mode="SYMMETRIC")
        
        print(noisy_img)
        print(weights)

        return weight_avg

    # Calculates the Peak Signal-to-noise value between two images
    def psnr(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    def kernelPredictMAV(self, y_true, y_pred):
        exp = tf.math.exp(y_pred)
        weight_sum = tf.reduce_sum(exp, axis=3, keepdims=True)
        weight_avg = tf.divide(exp, weight_sum)

        reference_img = tf.slice(y_true, (0, 0, 0, 0), (self.batch_size, config.PATCH_HEIGHT, config.PATCH_WIDTH, 3)) #y_true[:,:,:,0:3]
        noisy_img = tf.slice(y_true, (0, 0, 0, 3), (self.batch_size, config.PATCH_HEIGHT, config.PATCH_WIDTH, 3)) #y_true[:,:,:,0:3]

        kernel_radius = int(math.floor(self.kpcn_size / 2.0))
        paddings = tf.constant([[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])
        noisy_img = tf.pad(noisy_img, paddings, mode="SYMMETRIC")
    
        y_pred = tf.reshape(y_pred, shape=[self.batch_size, config.PATCH_HEIGHT, config.PATCH_WIDTH, self.kpcn_size, self.kpcn_size])

        new_noisy_img = tf.zeros((self.batch_size, config.PATCH_HEIGHT, config.PATCH_WIDTH, 3))

        #pred = weighted_average

        for pixel_x in range(config.PATCH_WIDTH):
            for pixel_y in range(config.PATCH_HEIGHT):
                accum = [0] * self.batch_size
                for kernel_x in range(-kernel_radius, kernel_radius):
                    for kernel_y in range(-kernel_radius, kernel_radius):
                        accum += noisy_img[:, pixel_x + kernel_x, pixel_y + kernel_y, 0] * \
                                 y_pred[:, pixel_x, pixel_y, kernel_x + kernel_radius, kernel_y + kernel_radius, 0]
                print(accum)
                #new_noisy_img[:, pixel_x, pixel_y, 0] =  0


        print(reference_img)
        print(noisy_img)
        print(y_true)
        print(y_pred)

        return tf.keras.losses.mean_squared_error(reference_img, y_pred)

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

        # 0.2 rescales to be similar to MSE loss values
        feature_loss = 0.03 * tf.reduce_sum(feature_loss) / (feature_shape[0] * feature_shape[1])

        return feature_loss

    def discriminatorLoss(self, y_pred):
        # Get the layers of discriminator model
        layers = [l for l in self.discriminator.model.layers]

        # Construct a graph to evaluate discriminator on y_pred
        eval_pred = y_pred
        for i in range(len(layers)):
            eval_pred = layers[i](eval_pred)

        discrim_loss = config.TRAIN_SCENES * config.NUM_DARTS * -tf.math.log(eval_pred)

        return discrim_loss

    def perceptualLoss(self, y_true, y_pred):
        feature_loss = self.VGG19FeatureLoss(y_pred, y_true)
        mse_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y_pred, y_true))

        mse_loss = tf.divide(mse_loss, config.PATCH_WIDTH * config.PATCH_HEIGHT)

        final_loss = tf.math.add(feature_loss, mse_loss)

        return final_loss

    def dropoutLayer(self, rate):
        self.model.add(tf.keras.layers.Dropout(rate))

    def buildNetwork(self):
        if self.batch_norm:
            self.trainBatchNorm()
        else:
            #self.initialConvLayer()
            #for _ in range(7):
            #    self.convLayer()
                #self.dropoutLayer(0.1)
            #self.finalConvLayer()

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
            x = self.returnFinalConvLayer(x)

            if self.kernel_predict:
                pred = tf.keras.layers.Lambda(
                    self.kernelPrediction, 
                    arguments={"noisy_img":conv_input}
                )(x)

            self.model = tf.keras.models.Model(inputs=conv_input, outputs=x)
            self.model.compile(
                optimizer=self.adam,
                loss=self.VGG19FeatureLoss,
                metrics=[self.psnr]
            )

    def train(self):
        if self.mse_epochs > 0:
            if self.kernel_predict:
                loss = self.kernelPredictMAV
            else:
                loss = "mean_absolute_error"

            self.model.compile(
                optimizer=self.adam,
                loss=loss,
                metrics=[self.psnr]
            )

            self.model.fit(
                self.train_input,
                self.train_labels,
                validation_data=(self.test_input, self.test_labels),
                batch_size=self.batch_size,
                epochs=self.mse_epochs,
                callbacks=self.callbacks
            )

            self.model.save("models/before_perceptual_loss")

        if self.vgg_epochs > 0:
            self.model.compile(
                optimizer=self.adam,
                loss=self.perceptualLoss,
                metrics=[self.psnr]
            )

            self.model.fit(
                self.train_input,
                self.train_labels,
                validation_data=(self.test_input, self.test_labels),
                batch_size=self.batch_size,
                epochs=self.vgg_epochs,
                callbacks=self.callbacks
            )

            self.model.save("models/after_perceptual_loss")
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


class GAN():
    """


    """
    def __init__(self, train_data, test_data, **kwargs):
        # --- Data hyperparameters --- #

        # The data dictionary
        self.train_data = train_data
        self.test_data = test_data

        self.num_epochs = kwargs.get("num_epochs", 10)

        # The adam optimiser is used, this block defines its parameters
        self.adam_lr = kwargs.get("adam_lr", 1e-4)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.999)
        self.adam_lr_decay = kwargs.get("adam_lr_decay", 0.0)

        self.adam = tf.keras.optimizers.Adam(
            lr=self.adam_lr,
            beta_1=self.adam_beta1,
            beta_2=self.adam_beta2,
            decay=self.adam_lr_decay
        )

        self.batch_size = kwargs.get("batch_size", 25)
        
    def psnr(self, y_true, y_pred):
        print(y_true)
        #print(y_pred)
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    def buildNetwork(self):
        self.discriminator.model.trainable = False
        gan_input = tf.keras.layers.Input((config.PATCH_HEIGHT, config.PATCH_WIDTH, self.denoiser.input_channels))
        denoiser_output = self.denoiser.model(gan_input)
        discrim_output = self.discriminator.model(denoiser_output)
        self.model = tf.keras.models.Model(inputs=gan_input, outputs=[denoiser_output, discrim_output])
        self.model.compile(
            loss=[self.denoiser.perceptualLoss, "binary_crossentropy"],
            #loss_weights=[1.0, 1e-3],
            loss_weights=[1.0, 1e-2],
            #loss_weights=[0, 1],
            optimizer=self.adam
            #metrics=[self.psnr]
        )
        

    def preTrainMAVDenoiser(self):
        feature_list = ["sn", "albedo", "depth"]
        mav_denoiser = Denoiser(
            self.train_data,
            self.test_data,
            mse_epochs=20,
            vgg_epochs=0,
            kernel_predict=True,
            feature_list=feature_list
        )
        if os.path.isfile("models/mav_denoiser"):
            print("Loading in initial mav denoiser...")
            mav_denoiser.model = tf.keras.models.load_model(
                "models/mav_denoiser",
                custom_objects={"psnr" : mav_denoiser.psnr}
            )
            mav_denoiser.eval(True)
        else:
            print("No mav_denoiser found - training now...")
            mav_denoiser.buildNetwork()
            mav_denoiser.train()
            mav_denoiser.eval(True)
            mav_denoiser.model.save("models/mav_denoiser")

        self.mav_denoiser = mav_denoiser
        self.denoiser = mav_denoiser

    def buildDenoiser(self):
        feature_list = ["sn", "albedo", "depth"]
        denoiser = Denoiser(
            self.train_data,
            self.test_data,
            mse_epochs=0,
            vgg_epochs=0,
            kernel_predict=True,
            feature_list=feature_list
        )
        denoiser.buildNetwork()
        self.denoiser = denoiser


    def buildDiscriminator(self):
        discriminator = Discriminator(
            self.train_data,
            self.test_data,
        )
        discriminator.buildNetwork()
        self.discriminator = discriminator


    # Train the discriminator on the output of the mean absolute value denoiser
    def preTrainDiscriminator(self, mav_train_pred, mav_test_pred):
        discriminator = Discriminator(
            self.train_data,
            self.test_data,
            mav_test_pred,
            num_epochs=1000,
        )

        if os.path.isfile("models/discriminator"):
            print("Loading in discriminator...")
            discriminator.model = tf.keras.models.load_model("models/discriminator") 
            discriminator.eval()
        else:
            print("No discriminator found - training now...")
            discriminator.buildNetwork()
            #discriminator.train()
            discriminator.eval()
            discriminator.model.save("models/discriminator")

        self.discriminator = discriminator

    def denoiserPredict(self):
        print("Evaluating denoiser on train data...")
        train_pred = self.denoiser.predict(self.train_data)
        print("Evaluating denoiser on test data...")
        test_pred = self.denoiser.predict(self.test_data)
        return train_pred, test_pred

    def train(self):
        #self.denoiser.discriminator = self.discriminator
        
        gan_writer = tf.summary.FileWriter("logs/gan-{}".format(time()))
        gan_val_writer = tf.summary.FileWriter("logs/gan_val-{}".format(time()), max_queue=1)
        adversarial_writer = tf.summary.FileWriter("logs/adversarial-{}".format(time()))
        #discrim_writer = tf.summary.FileWriter("logs/discriminator-{}".format(time()))

        self.denoiser.mse_epochs = 0
        self.denoiser.vgg_epochs = 1
        self.denoiser.vgg_mode = 54
        self.discriminator.num_epochs = 1
        train_data_size = config.TRAIN_SCENES * config.NUM_DARTS
        global_step = 0
        for epoch in range(self.num_epochs):
            print ('='*15, 'Epoch %d' % epoch, '='*15)
            for _ in tqdm(range(train_data_size // self.batch_size)):

                # Get random numbers to select our batch
                rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)

                # Get a batch and denoise
                train_noisy_batch = self.denoiser.train_input[rand_indices]
                train_reference_batch = np.array(self.denoiser.train_data["reference_colour"])[rand_indices]

                denoised_images = self.denoiser.model.predict(train_noisy_batch)

                # Create labels for the discriminator
                train_reference_labels = np.ones(self.batch_size)
                train_noisy_labels = np.zeros(self.batch_size)

                # Label smoothing
                train_reference_labels = train_reference_labels - np.random.random_sample(self.batch_size) * 0.2
                train_noisy_labels = train_noisy_labels + np.random.random_sample(self.batch_size) * 0.2

                # Unfreeze the discriminator so that we can train it
                self.discriminator.model.trainable = True

                discrim_input = np.concatenate((train_reference_batch, train_noisy_batch[:,:,:,0:3]), axis=0)
                discrim_labels = np.concatenate((train_reference_labels, train_noisy_labels), axis=0)

                discrim_loss_real = self.discriminator.model.train_on_batch(
                    train_reference_batch, 
                    train_reference_labels
                )

                discrim_loss_fake = self.discriminator.model.train_on_batch(
                    train_noisy_batch[:,:,:,0:3],
                    train_noisy_labels
                )

                rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)
                train_noisy_batch = self.denoiser.train_input[rand_indices]
                train_reference_batch = np.array(self.denoiser.train_data["reference_colour"])[rand_indices]

                # Create labels for the gan
                gan_labels = np.ones(self.batch_size)

                # Label smoothing
                gan_labels = gan_labels - np.random.random_sample(self.batch_size) * 0.2

                # Freeze the weights so the discriminator isn't trained with the denoiser in the gan network
                self.discriminator.model.trainable = False
                gan_loss = self.model.train_on_batch(train_noisy_batch, [train_reference_batch, gan_labels])

                # Add the discriminator's loss to tensorboard summary
                discrim_loss_real_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="discrim_loss_real", simple_value=discrim_loss_real[0]),]
                )

                discrim_acc_real_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="discrim_acc_real", simple_value=discrim_loss_real[1]),]
                )

                discrim_loss_fake_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="discrim_loss_fake", simple_value=discrim_loss_fake[0]),]
                )

                discrim_acc_fake_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="discrim_acc_fake", simple_value=discrim_loss_fake[1]),]
                )

                # Add the gan's loss to tensorboard summary
                gan_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="gan_loss", simple_value=gan_loss[0]),]
                )

                # Add the adversarial loss to tensorboard sumamry
                adversarial_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="adversarial_loss", simple_value=gan_loss[2]),]
                )
                    
                #discrim_writer.add_summary(discrim_acc_real_summary, global_step)
                #discrim_writer.add_summary(discrim_acc_fake_summary, global_step)
                #discrim_writer.add_summary(discrim_loss_real_summary, global_step)
                #discrim_writer.add_summary(discrim_loss_fake_summary, global_step)
                gan_writer.add_summary(gan_summary, global_step)
                adversarial_writer.add_summary(adversarial_summary, global_step)
                global_step += 1

            print("discriminator_loss_real :" + str(discrim_loss_real[0]))
            print("discriminator_loss_fake :" + str(discrim_loss_fake[0]))
            print("gan_loss :" + str(gan_loss[0]))
            if (epoch % 50) == 0:
                self.denoiser.model.save(self.denoiser.model_dir + "epoch:" + str(epoch))

            score = self.eval()
            gan_test_summary = tf.Summary(
                value=[tf.Summary.Value(tag="gan_loss", simple_value=score),]
            )
            gan_val_writer.add_summary(gan_test_summary, global_step)

            psnr = self.denoiser.eval(False)[1]
            psnr_summary = tf.Summary(
                value=[tf.Summary.Value(tag="psnr", simple_value=psnr),]
            )
            gan_writer.add_summary(psnr_summary, global_step)

            print("psnr :" + str(psnr))


    def eval(self):
        reference_test_images = np.array(self.test_data["reference_colour"])
        reference_test_labels = np.ones(len(reference_test_images))
        score = self.model.evaluate(self.denoiser.test_input, [reference_test_images, reference_test_labels], verbose=0)
        #print(" ")
        #print(" ===== GAN EVALUATION ===== ")
        print("gan_val_loss: " + str(score[0]))
        #print(" ")
        return score[0]


def denoise():
    seed = 1234
    patches = data.makePatches(seed)
    train_data = patches["train"]
    test_data = patches["test"]

    gan = GAN(train_data, test_data, num_epochs=1000)
    #gan.preTrainMAVDenoiser()
    gan.buildDenoiser()
    gan.buildDiscriminator()
    gan.buildNetwork()
    gan.train()
