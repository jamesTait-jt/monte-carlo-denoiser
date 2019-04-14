import math
from functools import partial

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import array_to_img, img_to_array
import keras
import keras.backend as K
from keras.layers.merge import _Merge

import config
import data
from denoiser import Denoiser
from discriminator import Discriminator
import weighted_average

# https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class GAN():
    """


    """
    def __init__(self, train_data, test_data, **kwargs):

        # --- Data hyperparameters --- #
        # The data dictionary
        self.train_data = train_data
        self.test_data = test_data

        self.denoiser_input_shape = (config.PATCH_WIDTH, config.PATCH_HEIGHT, 27)
        self.img_shape = (config.PATCH_WIDTH, config.PATCH_HEIGHT, 3)

        # What features are we going to use?
        self.feature_list = kwargs.get("feature_list", ["normal", "depth", "albedo"])

        # Prune the data to remove any features we don't want
        self.setInputAndLabels()

        # Set where we will save our model and logs
        self.model_dir = kwargs.get("model_dir", "default_gan")

        # --- Network hyperparameters --- #
        # Set the number of epochs
        self.num_epochs = kwargs.get("num_epochs", 100)

        # Set the batch size
        self.batch_size = kwargs.get("batch_size", 64)

        # set the size of the weights kernel
        self.kpcn_size = kwargs.get("kpcn_size", 21)

        # Set learning rates and adam params 
        self.g_lr = kwargs.get("g_lr", 1e-3)
        self.g_beta1 = kwargs.get("g_beta1", 0.9)
        self.g_beta2 = kwargs.get("g_beta2", 0.999)


        self.d_lr = kwargs.get("d_lr", 1e-4)

        self.c_lr = kwargs.get("c_lr", 1e-5)

        # How many times do we update critic per generator?
        self.c_itr = kwargs.get("c_itr", 10)

        # How much do we clip the weights of critic
        self.wgan_clip = kwargs.get("wgan_clip", 0.01)

        # Build and compile the Generator 
        self.generator = self.buildGenerator()
        #self.generator.summary()

        # Create our feature extractor for perceptual loss
        self.vgg = self.buildVGG()
        
        #self.compileVGG()
        #self.vgg.summary()

        # Build and compile the discriminator
        #self.discriminator = self.buildDiscriminator()
        #self.compileDiscriminator()
        #self.discriminator.summary()

        # Build and compile the critic
        self.critic = self.buildCritic()
        #self.compileCritic()
        #self.critic.summary()

        # Build (and compile inside) WGAN_GP model
        self.buildWGAN_GP()

        # Build and compile the combined GAN model
        #self.gan = self.buildGAN()
        #self.compileGAN()
        #self.gan.summary()

        # Build and compile wgan
        #self.wgan_gp = self.buildWGAN()
        #self.compileWGAN()
        #self.wgan.summary()

        # Initialise the instance noise parameter to 1
        self.discriminator_noise_parameter = 0.1

        # Set the number of update interations to 0
        self.global_step = 0

        # Set the model's timestamp
        self.timestamp = time()
        
    # ==== Data handling ==== #
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

    # ===== Functions to build the network ===== #
    def buildVGG(self):
        vgg19 = VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=self.img_shape
        )
        vgg19.trainable = False
        for layer in vgg19.layers:
            layer.trainable = False

        feature_extractor = keras.models.Model(
            inputs=vgg19.input,
            outputs=vgg19.get_layer("block2_conv2").output
        )

        feature_extractor.trainable = False
        return feature_extractor

    def buildGenerator(self):

        # Helper function for convolutional layer
        def convLayer(c_input, num_filters):
            c_output = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[5, 5],
                use_bias=True,
                strides=[1, 1],
                padding="SAME",
                kernel_initializer=keras.initializers.glorot_normal(seed=5678)
            )(c_input)
            return c_output

        ################################################

        # The generator takes a noisy image as input
        noisy_img = keras.layers.Input(self.denoiser_input_shape, name="Generator_input")

        # 9 fully convolutional layers
        x = convLayer(noisy_img, 100)
        x = keras.layers.ReLU()(x)
        for _ in range(7):
            x = convLayer(x, 100)
            #x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        # Final layer is not activated
        #weights = convLayer(x, pow(self.kpcn_size, 2))
        weights = convLayer(x, 3)

        return keras.models.Model(noisy_img, weights, name="Generator")

    def buildCritic(self):
        # Helper function for convolution layer
        def convBlock(c_input, num_filters, strides, bn=False):
            c_output = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                strides=strides,
                padding="SAME"
            )(c_input)
            
            output = keras.layers.LeakyReLU(alpha=0.2)(c_output)
            
            if bn:
                output = keras.layers.BatchNormalization(momentum=0.8)(output)
        
            return output

        ################################################

        img = keras.layers.Input(shape=self.img_shape, name="Critic_input")
        
        x = convBlock(img, 64, strides=[1, 1], bn=False)
        x = convBlock(x, 64, strides=[2, 2])
        x = convBlock(x, 128, strides=[1, 1])
        x = convBlock(x, 128, strides=[2, 2])
        x = convBlock(x, 256, strides=[1, 1])
        x = convBlock(x, 256, strides=[2, 2])
        #x = convBlock(x, 512, strides=[1, 1])
        #x = convBlock(x, 512, strides=[2, 2])

        x = keras.layers.Dense(1024)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1)(x)

        return keras.models.Model(img, x, name="Critic")

    def buildDiscriminator(self):

        # Helper function for convolution layer
        def convBlock(c_input, num_filters, strides, bn=True):
            c_output = keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                strides=strides,
                padding="SAME"
            )(c_input)
            
            output = keras.layers.LeakyReLU(alpha=0.2)(c_output)
            
            if bn:
                output = keras.layers.BatchNormalization(momentum=0.8)(output)
        
            return output

        ################################################

        img = keras.layers.Input(shape=self.img_shape, name="Discriminator_input")
        
        x = convBlock(img, 64, strides=[1, 1], bn=False)
        x = convBlock(x, 64, strides=[2, 2])
        x = convBlock(x, 128, strides=[1, 1])
        x = convBlock(x, 128, strides=[2, 2])
        x = convBlock(x, 256, strides=[1, 1])
        x = convBlock(x, 256, strides=[2, 2])
        x = convBlock(x, 512, strides=[1, 1])
        x = convBlock(x, 512, strides=[2, 2])

        x = keras.layers.Dense(1024)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Flatten()(x)
        prob = keras.layers.Dense(1, activation='sigmoid')(x)

        return keras.models.Model(img, prob, name="Discriminator")

    ################################################################
    #                HELPER FUNCTIONS - MAYBE SEPARATE             #
    ################################################################
    
    # Apply softmax function to normalise the weights
    def softmax(self, weights):
        # Subtract constant to avoid overflow
        weightmax = K.max(weights, axis=3, keepdims=True)
        weights = weights - weightmax

        exp = K.exp(weights)
        weight_sum = K.sum(exp, axis=3, keepdims=True)
        weights = tf.divide(exp, weight_sum)
        return weights

    # Prepare input data for weighted average function
    def processInputForKPCN(self, noisy_img):
        # Slice the noisy image out of the input
        noisy_img = noisy_img[:, :, :, 0:3]

        # Get the radius of the kernel
        kernel_radius = int(math.floor(self.kpcn_size / 2.0))

        # Pad the image on either side so that the kernel can reach all pixels
        paddings = tf.constant([[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])
        noisy_img = tf.pad(noisy_img, paddings, mode="SYMMETRIC")

        return noisy_img

    # Apply weighted average (KPCN)
    def applyKernel(self, noisy_img):
        import tensorflow as tf
        noisy_img = self.processInputForKPCN(noisy_img)
        def applyKernelLayer(weights):
            # Normalise the weights  (softmax)
            weights = self.softmax(weights)
            prediction = weighted_average.weighted_average(noisy_img, weights)
            return prediction
        return applyKernelLayer

    def buildWGAN_GP(self):

        # Freeze generator's layers while taining critic
        self.generator.trainable = False

        # Real (reference) image input
        real_img = keras.layers.Input(self.img_shape)

        # Noisy images input
        noisy_img = keras.layers.Input(self.denoiser_input_shape)

        # Get kernel of weights
        weights = self.generator(noisy_img)

        # Apply the weights to the noisy image
        #denoised_img = keras.layers.Lambda(self.applyKernel(noisy_img))(weights)
        denoised_img = weights

        # Normalise critic inputs between -1 and 1
        normalised_denoised_img = keras.layers.Lambda(
            lambda x: tf.image.per_image_standardization(x)
        )(denoised_img)
        
        normalised_real_img = keras.layers.Lambda(
            lambda x: tf.image.per_image_standardization(x)
        )(real_img)

        # Critic determines validity of real image and denoised image
        fake = self.critic(normalised_denoised_img)
        valid = self.critic(normalised_real_img)

        # Weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([normalised_real_img, normalised_denoised_img])
        
        # Critic determines validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Get loss function with averaged_samples argument 
        partial_gp_loss = partial(
            self.gradient_penalty_loss,
            averaged_samples=interpolated_img
        )
        
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = keras.models.Model(
            inputs=[real_img, noisy_img],
            outputs=[valid, fake, validity_interpolated]
        )

        self.critic_model.compile(
            loss=[self.wassersteinLoss, self.wassersteinLoss, partial_gp_loss],
            loss_weights=[1, 1, 10],
            optimizer=keras.optimizers.Adam(
                lr=self.c_lr,
                beta_1=0.5,
                beta_2=0.9
            )
        )

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Noisy image generator input
        noisy_img_gen = keras.layers.Input(self.denoiser_input_shape)

        # Generate images based of noise
        weights_gen = self.generator(noisy_img_gen)

        # Apply the weights to the noisy image
        #denoised_img_gen = keras.layers.Lambda(self.applyKernel(noisy_img_gen))(weights_gen)
        denoised_img_gen = weights_gen

        # Get the noisy features
        #denoised_features = self.vgg(
        #    denoised_img_gen
        #)
        #denoised_features = keras.layers.Lambda(lambda x: x, name='WGAN_GP_Feature')(denoised_features)

        # Critic determines validity
        valid = self.critic(denoised_img_gen)
        
        # Defines generator model
        self.generator_model = keras.models.Model(
            inputs=[noisy_img_gen],
            outputs=[denoised_img_gen, valid, denoised_img_gen]
        )

        self.generator_model.compile(
            loss=[self.featureLoss, self.wassersteinLoss, None],
            loss_weights=[1.0, 1.0, 0.0],
            optimizer=keras.optimizers.Adam(
                lr=self.g_lr,
                beta_1=self.g_beta1,
                beta_2=self.g_beta1
                #clipvalue=0.01
            )
        )

    def buildGAN(self):
        # Noisy images input
        noisy_img = keras.layers.Input(self.denoiser_input_shape, name="GAN_Input")

        # Denoise the images and get their features from the vgg network
        weights = self.generator(noisy_img)

        denoised_img = keras.layers.Lambda(self.applyKernel(noisy_img))(weights)

        denoised_features = self.vgg(
            denoised_img
        )

        # Freeze the weights of the discriminator since we only want to train
        # the generator
        self.discriminator.trainable = False

        # Run the denoised image through the discriminator
        denoised_probability = self.discriminator(denoised_img)

        # Create sensible names for outputs in logs
        denoised_features = keras.layers.Lambda(lambda x: x, name='Feature')(denoised_features)
        denoised_probability = keras.layers.Lambda(lambda x: x, name='Adversarial')(denoised_probability)

        # Create model using binary cross entropy with reversed labels
        
        return keras.models.Model(
            inputs=noisy_img,
            outputs=[denoised_probability, denoised_features, denoised_img], 
            name="GAN"
        )

    # ===== Functions to compile the models ===== #
    def compileVGG(self):
        adam = keras.optimizers.Adam(
            lr=0.0001,
            beta_1=0.9
        )
        self.vgg.compile(
            loss="mse",
            optimizer=adam,
            metrics=["accuracy"]
        )

    def compileGenerator(self):
        adam = keras.optimizers.Adam(
            lr=self.g_lr,
            beta_1=self.g_beta1,
            beta_2=self.g_beta2,
            #clipvalue=0.01
        )
        self.generator.compile(
            loss=self.featureLoss,
            optimizer=adam,
            metrics=["mse"]
        )
    
    #https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        # first get the gradients:
        #   assuming: - that y_pred has dimensions (batch_size, 1)
        #             - averaged_samples has dimensions (batch_size, nbr_features)
        # gradients afterwards has dimension (batch_size, nbr_features), basically
        # a list of nbr_features-dimensional gradient vectors
        gradients = K.gradients(y_pred, averaged_samples)[0]
        
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(
            gradients_sqr,
            axis=np.arange(1, len(gradients_sqr.shape))
        )
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def featureLoss(self, y_true, y_pred):
        features_true = self.vgg(y_true)
        features_pred = self.vgg(y_pred)
        feature_loss = keras.losses.mean_squared_error(features_pred, features_true)
        return K.mean(feature_loss)

    def wassersteinLoss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def compileCritic(self):
        adam = keras.optimizers.Adam(
            lr=self.c_lr,
            beta_1=0.5,
            beta_2=0.9
        )
        self.critic.compile(
            loss=self.wassersteinLoss,
            optimizer=adam
        )

    def compileDiscriminator(self):
        adam = keras.optimizers.Adam(
            lr=self.d_lr,
            beta_1=0.9,
            beta_2=0.999
        )
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=adam,
            metrics=["accuracy"]
        )

    def compileWGAN(self):
        adam = keras.optimizers.Adam(
            lr=self.g_lr,
            beta_1=0.5,
            beta_2=0.9
        )
        self.wgan.compile(
            loss=[self.wassersteinLoss, self.featureLoss, None], # FINAL LOSS IS JUST SO WE CAN HAVE 3 OUTPUTS
            loss_weights=[100, 0.1, 0], #Weighting from the paper
            optimizer=adam
        )

    def compileGAN(self):
        adam = keras.optimizers.Adam(
            lr=self.g_lr,
            beta_1=0.9,
            beta_2=0.999
        )
        self.gan.compile(
            loss=["binary_crossentropy", self.featureLoss, None], # FINAL LOSS IS JUST SO WE CAN HAVE 3 OUTPUTS
            loss_weights=[1, 0.006, 0], #Weighting from the paper
            #loss_weights=[1, 0, 0], #Weighting from the paper
            optimizer=adam
        )

    # We provide a function to train only thge generator. We must wrap the
    # generator so we can perform kernel prediction otherwise we will nto be
    # able to save the model.
    def trainGeneratorOnly(self, loss_function):

        # Build the generator with the wrapper to perform kernel predict
        def buildGeneratorWrapper(loss_function):

            # Noisy images input
            noisy_img = keras.layers.Input(self.denoiser_input_shape, name="GAN_Input")
            
            # Pass through the generator submodel to obtain weights
            weights = self.generator(noisy_img)

            # Apply the weights
            denoised_img = keras.layers.Lambda(self.applyKernel(noisy_img))(weights)

            # If mae, image is model output
            if loss_function == "mae":
                return keras.models.Model(
                    inputs=noisy_img,
                    outputs=denoised_img
                ) 
            
            # If feature loss, get them, and use as model output
            elif loss_function == "vgg":
                denoised_features = self.vgg(
                    denoised_img
                )
                denoised_features = keras.layers.Lambda(lambda x: x, name='Feature')(denoised_features)

                return keras.models.Model(
                    inputs=noisy_img,
                    outputs=[denoised_features, denoised_img]
                )
        
        def compileGeneratorWrapper(loss_function):
            adam = keras.optimizers.Adam(
                lr=self.g_lr,
                beta_1=0.9,
                beta_2=0.999,
            )
            if loss_function == "mae":
                self.generatorWrapper.compile(
                    loss="mean_absolute_error",
                    optimizer=adam
                )
            elif loss_function == "vgg":
                self.generatorWrapper.compile(
                    loss=["mean_squared_error", None],
                    optimizer=adam
                )

        def saveModel(loss_function):
            self.generator.save(
                "../models/generator_only/{0}-{1}-{2}".format(
                    self.timestamp, 
                    loss_function, 
                    self.g_lr
                )
            )

        def psnr(loss_function, epoch):
            reference_imgs = self.test_labels
            noisy_imgs = self.test_input

            if loss_function == "vgg":
                denoised = self.generatorWrapper.predict(noisy_imgs)[1]
            elif loss_function == "mae":
                denoised = self.generatorWrapper.predict(noisy_imgs)
        
            reference_imgs = np.clip(reference_imgs, 0, 1)
            denoised = np.clip(denoised, 0, 1)

            self.makeFigureAndSave(reference_imgs[0], denoised[0], epoch)

            reference_imgs_tensor = tf.placeholder(tf.float32, shape=reference_imgs.shape)
            denoised_tensor = tf.placeholder(tf.float32, shape=denoised.shape)

            psnr = tf.image.psnr(denoised_tensor, reference_imgs_tensor, max_val=1.0)
            #psnr = tf.where(tf.is_nan(psnr), tf.zeros_like(psnr), psnr)
            #psnr = tf.where(tf.is_inf(psnr), tf.zeros_like(psnr), psnr)
            #non_zeros = tf.count_nonzero(psnr)
            #psnr = tf.divide(K.sum(psnr), tf.cast(non_zeros, tf.float32))
            psnr = K.mean(psnr)

            with tf.Session("") as sess:
                inputs = {
                    reference_imgs_tensor : reference_imgs,
                    denoised_tensor : denoised
                }
                psnr = sess.run(psnr, feed_dict=inputs)

            return psnr

        ################################################

        self.generatorWrapper = buildGeneratorWrapper(loss_function)
        compileGeneratorWrapper(loss_function)

        train_data_size = np.array(self.train_data["noisy"]["diffuse"]).shape[0]
        test_data_size = np.array(self.test_data["noisy"]["diffuse"]).shape[0]
        step = 0

        train_writer = self.makeSummaryWriter("generator_only", loss_function, "train")
        val_writer = self.makeSummaryWriter("generator_only", loss_function, "val")

        for epoch in range(self.num_epochs):
            print("="*15, "Epoch %d" % epoch, "="*15)
            for batch_num in tqdm(range(train_data_size // self.batch_size)):

                rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)
                train_noisy_batch = self.train_input[rand_indices]
                train_reference_batch = self.train_labels[rand_indices]

                if loss_function == "vgg":
                    reference_features = self.vgg(
                        train_reference_batch
                    )
                    train_loss = self.generatorWrapper.train_on_batch(
                        train_noisy_batch,
                        reference_features
                    )[0]

                elif loss_function == "mae":
                    train_loss = self.generatorWrapper.train_on_batch(
                        train_noisy_batch,
                        train_reference_batch
                    )

                train_loss_summary = self.makeSummary("loss", train_loss)
                train_writer.add_summary(train_loss_summary, step)
                step += 1
            
            # Save the model at the end of each epoch
            #saveModel(loss_function)

            # Calculate the psnr on the test set
            val_psnr = psnr(loss_function, epoch)
            psnr_summary = self.makeSummary("psnr", val_psnr)
            val_writer.add_summary(psnr_summary, step)

            # Calculate the validation loss on a random batch
            rand_indices = np.random.randint(0, test_data_size, size=self.batch_size)
            if loss_function == "vgg":
                test_ref_features = self.vgg(
                    self.test_labels[rand_indices]
                )
                val_loss = self.generatorWrapper.test_on_batch(
                    self.test_input[rand_indices],
                    test_ref_features
                )[0]
            elif loss_function == "mae":
                val_loss = self.generatorWrapper.evaluate(
                    self.test_input,
                    self.test_labels
                )
            
            val_loss_summary = self.makeSummary("loss", val_loss)
            val_writer.add_summary(val_loss_summary, step)

            print("PSNR: %f" % val_psnr)
            print("VAL_LOSS: %f" % val_loss)


    # Convert from [0, 1] to [0, 255] for vgg
    def preprocessVGG(self, img):
        if isinstance(img, np.ndarray):
            return preprocess_input(img * 255.0)
        else:            
            return keras.layers.Lambda(lambda x: preprocess_input(x * 255))(img)

    def makeLabelsNoisy(pct, real_labels, fake_labels):
        for i in range(len(real_labels)):
            rand_num = np.random.random()
            if rand_num < pct:
                real_labels[i] = 0

        for i in range(len(fake_labels)):
            rand_num = np.random.random()
            if rand_num < pct:
                fake_labels[i] = 1#0.9 #np.random.uniform(0.9, 1.0)

        return real_labels,fake_labels

    def trainDiscriminator(self, real_batch, fake_batch):
       
        # Create labels for the discriminator
        #train_reference_labels = np.random.uniform(0.9, 1.0, size=self.batch_size)# One sided label smoothing
        real_labels = np.ones(self.batch_size) #* 0.9 # One sided label smoothing
        fake_labels = np.zeros(self.batch_size)
       
        # Add noise to discrim real input
        real_input_noise = np.random.normal(
            scale=self.discriminator_noise_parameter,
            size=(self.batch_size, config.PATCH_WIDTH, config.PATCH_HEIGHT, 3)
        )
        #real_batch += real_input_noise

        # Train discriminator on real batch
        loss_real = self.discriminator.train_on_batch(
            real_batch, 
            real_labels
        )

        # Add noise to discrim fake batch
        fake_input_noise = np.random.normal(
            scale=self.discriminator_noise_parameter,
            size=(self.batch_size, config.PATCH_WIDTH, config.PATCH_HEIGHT, 3)
        )
        #fake_batch += fake_input_noise

        # Train discriminator of fake batch
        loss_fake = self.discriminator.train_on_batch(
            fake_batch,
            fake_labels
        )
        real_acc = loss_real[1]
        fake_acc = loss_fake[1]
        real_loss = loss_real[0]
        fake_loss = loss_fake[0]
        acc = 0.5 * np.add(loss_real[1], loss_fake[1])
        loss = 0.5 * np.add(loss_real[0], loss_fake[0])

        return (real_acc, fake_acc, acc, real_loss, fake_loss, loss)

    def trainWGAN_GP(self):

        def saveModel(loss_function):
            self.generator.save(
                "../models/wgan-gp/{0}".format(
                    self.timestamp
                )
            )

        def psnr(epoch):
            reference_imgs = self.test_labels
            noisy_imgs = self.test_input

            denoised = self.generator_model.predict(noisy_imgs)[2]
        
            reference_imgs = np.clip(reference_imgs, 0, 1)
            denoised = np.clip(denoised, 0, 1)

            self.makeFigureAndSave(reference_imgs[0], denoised[0], epoch)

            reference_imgs_tensor = tf.placeholder(tf.float32, shape=reference_imgs.shape)
            denoised_tensor = tf.placeholder(tf.float32, shape=denoised.shape)

            psnr = tf.image.psnr(denoised_tensor, reference_imgs_tensor, max_val=1.0)
            psnr = tf.where(tf.is_inf(psnr), tf.zeros_like(psnr), psnr)
            non_zeros = tf.count_nonzero(psnr)
            psnr = tf.divide(K.sum(psnr), tf.cast(non_zeros, tf.float32))

            with tf.Session("") as sess:
                inputs = {
                    reference_imgs_tensor : reference_imgs,
                    denoised_tensor : denoised
                }
                psnr = sess.run(psnr, feed_dict=inputs)

            return psnr

        ########################################################################

        train_data_size = np.array(self.train_data["noisy"]["diffuse"]).shape[0]
        test_data_size = np.array(self.test_data["noisy"]["diffuse"]).shape[0]

        print("  ========================================== ")
        print(" || Training on %d Patches                 ||" % train_data_size)
        print(" || Testing on %d Patches                  ||" % test_data_size)
        print("  ========================================== ")

        real = np.ones((self.batch_size, 1))
        fake = -real
        dummy = real * 0.0 #Dummy labels of 0  
        
        val_writer = self.makeSummaryWriter("wgan-gp", "vgg+adv", "val")
        d_loss_writer = self.makeSummaryWriter("wgan-gp", "vgg+adv", "train")
        g_feat_loss_writer = self.makeSummaryWriter("wgan-gp", "vgg+adv", "train_feature")
        g_adv_loss_writer = self.makeSummaryWriter("wgan-gp", "vgg+adv", "train_adv")

        step = 0
        for epoch in range(self.num_epochs):
            print("="*15, "Epoch %d" % epoch, "="*15)
            for batch_num in tqdm(range(train_data_size // self.batch_size)):

                d_loss = [0]
                for _ in range(self.c_itr):
                    # Get random numbers to select our batch
                    rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)

                    # Get a batch and denoise
                    train_noisy_batch = self.train_input[rand_indices]
                    train_reference_batch = self.train_labels[rand_indices]

                    # Train critic
                    d_loss = self.critic_model.train_on_batch(
                        [train_reference_batch, train_noisy_batch], 
                        [real, fake, dummy]
                    )

                # Get random numbers to select our batch
                rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)

                # Get a batch and denoise
                train_noisy_batch = self.train_input[rand_indices]
                train_reference_batch = self.train_labels[rand_indices]
                
                # Train generator

                g_loss = self.generator_model.train_on_batch(
                    train_noisy_batch, 
                    [train_reference_batch, real]
                )

                feat_loss = g_loss[1]
                adv_loss = g_loss[2]

                d_loss_summary = self.makeSummary("d_loss", -d_loss[0])
                d_loss_writer.add_summary(d_loss_summary, step)

                g_feat_loss_summary = self.makeSummary("g_loss", 0.1 * feat_loss)
                g_feat_loss_writer.add_summary(g_feat_loss_summary, step)

                g_adv_loss_summary = self.makeSummary("g_loss", adv_loss)
                g_adv_loss_writer.add_summary(g_adv_loss_summary, step)
                
                step += 1

            # Save the model at each epoch
            saveModel()

            val_psnr = psnr(epoch)
            val_psnr_summary = self.makeSummary("psnr", val_psnr)
            val_writer.add_summary(val_psnr_summary, step)
            print("PSNR: %f" % val_psnr)
            print("D_LOSS: %f" % d_loss[0])
            print("G_ADV_LOSS: %f" % adv_loss)
                
    def train(self):
        #self.setTensorBoardWriters()

        train_data_size = np.array(self.train_data["noisy"]["diffuse"]).shape[0]
        test_data_size = np.array(self.test_data["noisy"]["diffuse"]).shape[0]

        print("  ========================================== ")
        print(" || Training on %d Patches                 ||" % train_data_size)
        print(" || Testing on %d Patches                  ||" % test_data_size)
        print("  ========================================== ")

        discrim_real_acc = 0
        discrim_fake_acc = 0
        discrim_loss = 0
        for epoch in range(self.num_epochs):
            print("="*15, "Epoch %d" % epoch, "="*15)
            epoch_discrim_acc = 0
            for batch_num in tqdm(range(train_data_size // self.batch_size)):
                discrim_acc = 1
                for i in range(discrim_acc):
                    # Get random numbers to select our batch
                    rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)
    
                    # Get a batch and denoise
                    train_noisy_batch = self.train_input[rand_indices]
                    train_reference_batch = self.train_labels[rand_indices]

                    denoised_batch = self.gan.predict(train_noisy_batch)[2].copy()
                    ref = train_reference_batch[0].copy()
    
                    (discrim_real_acc, discrim_fake_acc, discrim_acc, real_loss, fake_loss, discrim_loss) = \
                        self.trainDiscriminator(
                            train_reference_batch,
                            denoised_batch
                        )
                
                    #print("---------------")
                    #print(discrim_fake_acc)
                    #print(discrim_real_acc)
                    #print(discrim_acc)

                rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)
                train_noisy_batch = self.train_input[rand_indices]
                train_reference_batch = self.train_labels[rand_indices]
                    
                # Get the features for the reference images
                reference_features = self.vgg(
                    train_reference_batch
                )

                # Train the generator
                gen_loss = self.gan.train_on_batch(
                    train_noisy_batch,
                    [np.ones(self.batch_size), reference_features]
                ) 

                # Write batch-based summaries to tensorboard
                #self.makeBatchSummariesAndWrite(
                #    discrim_real_acc,
                #    discrim_fake_acc,
                #    discrim_loss,
                #    real_loss,
                #    fake_loss,
                #    gen_loss
                #)
            
                self.global_step += 1

            denoised = denoised_batch[0]
            self.makeFigureAndSave(ref, denoised, epoch)

            # Calculate the feature loss and psnr on a test batch
            #val_f_loss, val_psnr = self.denoiser.eval(False)
            val_psnr = self.testPSNR()

            # Write epoch-based summaries to tensorboard
            #self.makeEpochSummariesAndWrite(val_psnr)#, val_f_loss)
            
            # Decrease instance noise
            #if self.discriminator_noise_parameter > 0:
            #    self.discriminator_noise_parameter -= 0.01

            print("val_psnr: " + str(val_psnr))
            print("gen_feature_loss: " + str(gen_loss[2] * 0.006))
            print("gen_adversarial_loss: " + str(gen_loss[1] * 1))

            self.generator.save("../models/" + self.model_dir + "-{}".format(self.timestamp))
            #self.model.save("../models/gan-{}".format(self.timestamp))
            #if (epoch % 50) == 0:
            #    self.denoiser.model.save(self.denoiser.model_dir + "epoch:" + str(epoch))

    def makeFigureAndSave(self, ref, denoised, epoch):

        ref = ref.clip(0, 1)
        denoised = denoised.clip(0, 1)

        ref_img = array_to_img(ref)
        denoised_img = array_to_img(denoised)

        fig = plt.figure()

        ref_subplot = plt.subplot(121)
        ref_subplot.set_title("Reference image")
        ref_subplot.imshow(ref_img)
    
        denoised_subplot = plt.subplot(122)
        denoised_subplot.set_title("Denoised image")
        denoised_subplot.imshow(denoised_img)

        fig.savefig("../training_results/epoch:%d" % epoch)

    def testPSNR(self): 
        reference_test_imgs = self.test_labels
        noisy_test_imgs = self.test_input
        denoised = self.gan.predict(noisy_test_imgs)[2]
        
        denoised = np.clip(denoised, 0, 1)
        reference_test_imgs = np.clip(reference_test_imgs, 0, 1)

        with tf.Session(""):
            psnr = tf.image.psnr(denoised, reference_test_imgs, max_val=1.0)
            psnr = tf.where(tf.is_nan(psnr), tf.zeros_like(psnr), psnr)
            non_zeros = tf.count_nonzero(psnr)
            psnr = tf.divide(K.sum(psnr), tf.cast(non_zeros, tf.float32)).eval()
        return psnr
        
    def makeSummary(self, tag, val):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=val),]
        )
        return summary

    def makeSummaryWriter(self, directory, loss_function, name):
        writer = tf.summary.FileWriter(
            "../logs/{0}/{1}-{2}-{3}-{4}".format(
                directory,
                self.timestamp, 
                loss_function,
                self.g_lr,
                name
            ),
            max_queue=1,
            flush_secs=10
        )
        return writer

