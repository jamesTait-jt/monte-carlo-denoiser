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
import tensorflow as tf
from keras.preprocessing.image import array_to_img
import data
import config
import numpy as np
from time import time

########################
##### Global flags #####
########################
FLAGS = tf.app.flags.FLAGS

##### Network #####
tf.app.flags.DEFINE_integer ("patchSize", config.PATCH_WIDTH,
                            "The size of the input patches")

#tf.app.flags.DEFINE_integer ("reconstructionKernelSize", 21,
#                            "The size of the reconstruction kernel")

tf.app.flags.DEFINE_integer ("inputChannels", 27,
                            "The number of channels in an input patch")

tf.app.flags.DEFINE_integer ("outputChannels", 3,
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

        # General network hyperparameters
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

        # Set the log directory with a timestamp
        #self.log_dir = "logs/{}".format(time())
        self.set_log_dir()
        self.set_model_dir()
        
        # Use the sequential model API
        self.model = tf.keras.models.Sequential()

        # Set callbacks
        self.set_callbacks()

    def set_log_dir(self):
        log_dir = "logs/"
        for feature in self.feature_list:
            log_dir += (feature + "&")

        log_dir += ("lr:" + str(self.adam_lr) + "&")
        log_dir += ("lr_decay:" + str(self.adam_lr_decay) + "&")
        log_dir += ("bn:" + str(self.batch_norm) + "&")

        self.log_dir = log_dir + "{}".format(time())

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
        #self.callbacks.append(tensorboard_cb)

        tensorboard_cb = TrainValTensorBoard(log_dir=self.log_dir, write_graph=True)

        self.callbacks.append(tensorboard_cb)


    def set_input_and_output_data(self):
        new_train_in = [
            self.train_data["colour"]["noisy"], 
            self.train_data["colour_gradx"]["noisy"],
            self.train_data["colour_grady"]["noisy"],
            self.train_data["colour_var"]["noisy"]
        ] 

        new_test_in = [
            self.test_data["colour"]["noisy"], 
            self.test_data["colour_gradx"]["noisy"],
            self.test_data["colour_grady"]["noisy"],
            self.test_data["colour_var"]["noisy"]
        ]

        print(np.concatenate((new_train_in), 3).shape)

        for feature in self.feature_list:
            # Each feature is split into gradient in X and Y direction, and it's
            # corresponding variance
            feature_keys = [feature + "_gradx", feature + "_grady", feature + "_var"]
            for key in feature_keys:
                new_train_in.append(self.train_data[key]["noisy"])
                new_test_in.append(self.test_data[key]["noisy"])
            

        self.train_input = np.concatenate((new_train_in), 3)
        self.test_input = np.concatenate((new_test_in), 3)
        
        self.train_output = self.train_data["colour"]["reference"]
        self.test_output = self.test_data["colour"]["reference"]

        # Ensure input channels is the right size
        self.input_channels = self.train_input.shape[3]
        print(self.train_input.shape)

    # First convolutional layer (must define input shape)
    def initialConvLayer(self):
        self.model.add(
            tf.keras.layers.Conv2D(
                input_shape=(self.patch_height, self.patch_width, self.input_channels),
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                use_bias=True,
                strides=(1, 1),
                padding="SAME",
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
                padding="SAME",
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
                padding="SAME",
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
        self.model.add(
            tf.keras.layers.Conv2D(
                filters=self.output_channels,
                kernel_size=self.kernel_size,
                use_bias=True,
                strides=(1, 1),
                padding="SAME",
                activation=None,
                kernel_initializer="glorot_uniform" # Xavier uniform
            )
        )

    def trainBatchNorm(self):
        for i in range(8):
            self.convWithBatchNorm()
        self.finalConvLayer()

    def train(self):

        if self.batch_norm:
            self.trainBatchNorm()

        else:
            self.initialConvLayer()
            for i in range(14):
                self.convLayer()
            self.finalConvLayer()

        self.model.compile(
            optimizer=self.adam,
            loss="mean_absolute_error",
            metrics=["accuracy"]
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
    train_data = data.data["train"]
    test_data = data.data["test"]

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
        num_epochs=10,
        adam_lr=1e-5,
        batch_norm=False,
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
