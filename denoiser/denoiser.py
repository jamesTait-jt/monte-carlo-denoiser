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
import keras

class Discriminator():
    """Class to discriminate between real and fake reference images
    
    This class is trained to recognize reference (not noisy) images, and
    determine whether an image passed in is a real reference image, or a fake
    one

    """

    def __init__(
        self, 
        train_ref_images,
        train_mav_pred,
        test_ref_images,
        test_mav_pred,
        **kwargs
    ):
        # --- Data hyperparameters --- #

        # Data dictionary
        self.train_ref_images = train_ref_images
        self.train_mav_pred = train_mav_pred

        self.test_ref_images = test_ref_images
        self.test_mav_pred = test_mav_pred
        self.setInputAndOutputData()

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
        train_fake_labels = np.zeros([len(self.train_mav_pred), 1])

        self.train_input = np.concatenate((self.train_ref_images, self.train_mav_pred))
        self.train_labels = np.concatenate((train_reference_labels, train_fake_labels)) 

        test_reference_labels = np.ones([len(self.test_ref_images), 1])
        test_fake_labels = np.zeros([len(self.test_mav_pred), 1])

        self.test_input = np.concatenate((self.test_ref_images, self.test_mav_pred))
        self.test_labels = np.concatenate((test_reference_labels, test_fake_labels)) 

    def setCallbacks(self):
        self.callbacks = []

        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )

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
            #min_delta=0.02
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

    def denseLayer(self, units):
        self.model.add(tf.keras.layers.Dense(units))

    def sigmoid(self):
        self.model.add(tf.keras.layers.Activation("sigmoid"))

    def flatten(self):
        self.model.add(tf.layers.Flatten())

    def initNetwork(self):
        self.initialConvLayer(3, 64, [1, 1]) 

        self.discriminatorBlock(3, 64 , [2, 2])
        self.discriminatorBlock(3, 128, [1, 1])
        self.discriminatorBlock(3, 128, [2, 2])
        self.discriminatorBlock(3, 256, [1, 1])
        self.discriminatorBlock(3, 256, [2, 2])
        self.discriminatorBlock(3, 512, [1, 1])
        self.discriminatorBlock(3, 512, [2, 2])

        self.denseLayer(1024)
        self.leakyReLU()
        self.flatten()
        self.denseLayer(1)
        self.sigmoid()

    def train(self):
        self.model.compile(
            optimizer=self.adam,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

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
        self.set_input_and_output_data()

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
        if (self.vgg_epochs > 0):
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
        self.kpcn_size = kwargs.get("kpcn_size", 20)

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

    def VGG19FeatureLoss(self, y_pred, y_true):
        vgg19 = VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg19.trainable = False
        for layer in vgg19.layers:
            layer.trainable = False

        if self.vgg_mode == 54:
            feature_extractor = tf.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer("block5_conv4").output)
            paddings = tf.constant([[0, 0], [0, 60], [0, 60]])
            mode = "CONSTANT"
            feature_shape = [4, 4]
        elif self.vgg_mode == 22:
            feature_extractor = tf.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer("block2_conv2").output)
            paddings = tf.constant([[0, 0], [0, 32], [0, 32]])
            mode = "SYMMETRIC"
            feature_shape = [32, 32]

        features_pred = feature_extractor(y_pred)
        features_true = feature_extractor(y_true)

        feature_loss = tf.keras.losses.mean_squared_error(features_pred, features_true)

        # 0.006 rescales to be similar to MSE loss values
        feature_loss = 0.06 * tf.reduce_sum(feature_loss) / (feature_shape[0] * feature_shape[1])

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
        discrim_loss = self.discriminatorLoss(y_pred)

        final_loss = tf.math.add(feature_loss, 1e-3 * discrim_loss)

        return final_loss

    def dropoutLayer(self, rate):
        self.model.add(tf.keras.layers.Dropout(rate))

    def initNetwork(self):
        if self.batch_norm:
            self.trainBatchNorm()

        else:
            self.initialConvLayer()
            for _ in range(7):
                self.convLayer()
                #self.dropoutLayer(0.1)
            self.finalConvLayer()

    def train(self):

        if self.mse_epochs > 0:
            self.model.compile(
                optimizer=self.adam,
                loss="mean_absolute_error",
                metrics=[self.psnr]
            )

            self.model.fit(
                self.train_input,
                self.train_output,
                validation_data=(self.test_input, self.test_output),
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
                self.train_output,
                validation_data=(self.test_input, self.test_output),
                batch_size=self.batch_size,
                epochs=self.vgg_epochs,
                callbacks=self.callbacks
            )

            self.model.save("models/after_perceptual_loss")
        self.model.save(self.model_dir)

    def predict(self, data):
        model_input = []
        for key in data:
            if not key == "reference_colour":
                model_input.append(np.array(data[key]))
        model_input = np.concatenate((model_input), 3)
        pred = self.model.predict(model_input)
        return pred

    def eval(self):
        score = self.model.evaluate(self.test_input, self.test_output, verbose=0)
        print(" ")
        print(" ===== DENOISER EVALUATION ===== " )
        print(" ==== Test loss: " + str(score[0]) + " ==== ")
        print(" ==== Test PSNR: " + str(score[1]) + " ==== ")
        print(" ")

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


def preTrainMAVDenoiser(train_data, test_data):
    feature_list = ["sn", "albedo", "depth"]
    mav_denoiser = Denoiser(
        train_data, 
        test_data, 
        mse_epochs=1000,
        vgg_epochs=0,
        feature_list=feature_list
    )
    if os.path.isfile("models/mav_denoiser"):
        print("Loading in initial mav denoiser...")
        mav_denoiser.model = tf.keras.models.load_model(
            "models/mav_denoiser", 
            custom_objects={"psnr" : mav_denoiser.psnr}
        ) 
        mav_denoiser.eval()
    else:
        print("No mav_denoiser found - training now...")
        mav_denoiser.initNetwork()
        mav_denoiser.train()
        mav_denoiser.eval()
        mav_denoiser.model.save("models/mav_denoiser")

    return mav_denoiser


# Train the discriminator on the output of the mean absolute value denoiser
def preTrainDiscriminator(train_ref_colour_patches, mav_train_pred, test_ref_colour_patches, mav_test_pred):
    discriminator = Discriminator(
        train_ref_colour_patches,
        mav_train_pred,
        test_ref_colour_patches,
        mav_test_pred,
        num_epochs=1000,
    )

    if os.path.isfile("models/discriminator"):
        print("Loading in discriminator...")
        discriminator.model = tf.keras.models.load_model("models/discriminator") 
        discriminator.eval()
    else:
        print("No discriminator found - training now...")
        discriminator.initNetwork()
        discriminator.train()
        discriminator.eval()
        discriminator.model.save("models/discriminator")

    return discriminator

def denoise():

    seed = 1234
    patches = make_patches.makePatches(seed)
    train_data = patches["train"]
    test_data = patches["test"]

    # Pre train a mean absolute value denoiser
    mav_denoiser = preTrainMAVDenoiser(train_data, test_data)

    # Evaluate the MAV denoiser on training data for input to discriminator
    print("Evaluating pre trained MAV denoiser on train data...")
    train_pred = mav_denoiser.predict(train_data)
    print("Evaluating pre trained MAV denoiser on test data...")
    test_pred = mav_denoiser.predict(test_data)

    # Extract the colour patches from the data ready for the discriminator
    train_ref_colour_patches = train_data["reference_colour"]
    train_noisy_colour_patches = train_data["noisy_colour"]
    test_ref_colour_patches = test_data["reference_colour"]
    test_noisy_colour_patches = test_data["noisy_colour"]

    # Pre train discriminator to use in denoiser's loss function
    discriminator = preTrainDiscriminator(
        train_ref_colour_patches,
        train_pred,
        test_ref_colour_patches,
        test_pred
    )
    del train_pred
    del test_pred

    # Set denoiser to mav_denoiser to get the model for the first iteration
    denoiser = mav_denoiser
    del mav_denoiser

    for i in range(100):

        denoiser = Denoiser(
            train_data, 
            test_data, 
            discriminator=discriminator,
            model=denoiser.model,    # We restart from the MAV trained model to avoid undesired local optima
            mse_epochs=00,
            vgg_epochs=1,
            vgg_mode=54,
            feature_list=denoiser.feature_list
        )
        denoiser.train()
        denoiser.eval()

        # Evaluate the MAV denoiser on training data for input to discriminator
        print("Evaluating new denoiser on train data...")
        train_pred = denoiser.predict(train_data)
        print("Evaluating new denoiser on test data...")
        test_pred = denoiser.predict(test_data)

        print(" ======================================== ")
        print(" === Denoising iteration: " + str(i) + " complete === ")
        print(" ======= Retraining discriminator =======  ")
        print(" ======================================== ")

        discriminator = Discriminator(
            train_ref_colour_patches,
            train_pred,
            test_ref_colour_patches,
            test_pred,
            model=discriminator.model,
            num_epochs=1,
        )
        discriminator.train()
        discriminator.eval()


    del denoiser
    del discriminator
