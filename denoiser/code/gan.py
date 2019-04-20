import math
from functools import partial

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import array_to_img, img_to_array
import keras
import keras.backend as K
from keras.layers.merge import _Merge

import config
import models
import data
from denoiser import Denoiser
from discriminator import Discriminator
import weighted_average

# https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1, 1))
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
        self.log_dir = kwargs.get("log_dir", "default_gan")

        # --- Network hyperparameters --- #
        
        # Are we doing KPCN or DPCN
        self.kernel_predict = kwargs.get("kernel_predict", False)

        # Set the number of epochs
        self.num_epochs = kwargs.get("num_epochs", 100)

        # Set the batch size
        self.batch_size = kwargs.get("batch_size", 64)

        # set the size of the weights kernel
        self.kpcn_size = kwargs.get("kpcn_size", 21)

        g_loss = kwargs.get("g_loss", self.featureLoss)
        if g_loss == "mae":
            self.g_loss = "mean_absolute_error"
        elif g_loss == "vgg":
            self.g_loss = self.featureLoss

        # Weighting of the loss function
        self.loss_weights = kwargs.get("loss_weights", [1, 1])
        

        # Set learning rates and adam params 
        self.g_lr = kwargs.get("g_lr", 1e-3)
        self.g_beta1 = kwargs.get("g_beta1", 0.9)
        self.g_beta2 = kwargs.get("g_beta2", 0.999)

        # How many hidden layers for generator
        self.g_layers = kwargs.get("g_layers", 6)


        # Kernel size for generator
        self.g_kernel_size = kwargs.get("g_kernel_size", [3, 3])

        # Do we do batch norm on the generator
        self.g_bn = kwargs.get("g_bn", False)

        # Set learning rate for discriminator
        self.d_lr = kwargs.get("d_lr", 1e-4)

        # Set learning rate for critic
        self.c_lr = kwargs.get("c_lr", 1e-5)

        # How many times do we update critic per generator?
        self.c_itr = kwargs.get("c_itr", 5)

        # How much do we clip the weights of critic
        self.wgan_clip = kwargs.get("wgan_clip", 0.01)

        # Build and compile the Generator 
        self.generator = models.buildGenerator(
            self.g_kernel_size,
            self.g_layers,
            self.g_bn,
            self.kernel_predict,
            self.kpcn_size
        )
        self.generator.summary()

        # Create our feature extractor for perceptual loss
        self.vgg = models.buildVGG()
        
        #self.compileVGG()
        #self.vgg.summary()

        # Build and compile the discriminator
        #self.discriminator = self.buildDiscriminator()
        #self.compileDiscriminator()
        #self.discriminator.summary()

        # Build and compile the critic
        self.critic = models.buildCritic()
        #self.compileCritic()
        self.critic.summary()

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
        self.timestamp = str(time())
        
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
        
        # If we are using the critic's loss
        if self.loss_weights[1] > 0:
            # Freeze generator's layers while taining critic
            self.generator.trainable = False

            # Real (reference) image input
            real_img = keras.layers.Input(self.img_shape)

            # Noisy images input
            noisy_img = keras.layers.Input(self.denoiser_input_shape)


            # Apply the weights to the noisy image
            if self.kernel_predict:
                # Get kernel of weights
                weights = self.generator(noisy_img)
                denoised_img = keras.layers.Lambda(self.applyKernel(noisy_img))(weights)
            else:
                denoised_img = self.generator(noisy_img)

            # Critic determines validity of real image and denoised image
            fake = self.critic(denoised_img)
            valid = self.critic(real_img)

            # Weighted average between real and fake images
            interpolated_img = RandomWeightedAverage()([real_img, denoised_img])
            
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

        if self.kernel_predict:
            # Generate images based of noise
            weights_gen = self.generator(noisy_img_gen)
            # Apply the weights to the noisy image
            denoised_img_gen = keras.layers.Lambda(self.applyKernel(noisy_img_gen))(weights_gen)
        else:
            denoised_img_gen = self.generator(noisy_img_gen)

        # Critic determines validity
        valid = self.critic(denoised_img_gen)
        
        # Defines generator model
        self.generator_model = keras.models.Model(
            inputs=[noisy_img_gen],
            outputs=[denoised_img_gen, valid]
        )

        self.generator_model.compile(
            loss=[self.g_loss, self.wassersteinLoss],
            #loss_weights=[0.1, 1.0],
            loss_weights=self.loss_weights,
            optimizer=keras.optimizers.Adam(
                lr=self.g_lr,
                beta_1=self.g_beta1,
                beta_2=self.g_beta1,
                clipvalue=1
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
            clipvalue=1
        )
        self.generator.compile(
            loss=self.featureLoss,
            optimizer=adam
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
        gradient_penalty = K.square(gradient_l2_norm)
        
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

        def saveModel(model):
            model.save(self.model_dir + self.timestamp)

        def psnr(epoch):
            reference_imgs = self.test_labels
            noisy_imgs = self.test_input

            denoised = self.generator_model.predict(noisy_imgs)[0]
        
            if config.ALBEDO_DIVIDE:
                reference_imgs = np.array(self.test_data["reference"]["diffuse"])
                denoised *= (np.array(self.test_data["noisy"]["albedo"]) + 0.00316)

            reference_imgs = np.clip(reference_imgs, 0, 1)
            denoised = np.clip(denoised, 0, 1)

            rnd = np.random.randint(noisy_imgs.shape[0])
            self.makeFigureAndSave(reference_imgs[rnd], denoised[rnd], epoch)

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
        dummy = np.zeros((self.batch_size)) #Dummy labels of 0  
        
        val_writer = self.makeSummaryWriter("wgan-gp", "vgg+adv", "val")
        d_loss_writer = self.makeSummaryWriter("wgan-gp", "vgg+adv", "train")
        g_feat_loss_writer = self.makeSummaryWriter("wgan-gp", "vgg+adv", "train_feature")
        g_adv_loss_writer = self.makeSummaryWriter("wgan-gp", "vgg+adv", "train_adv")

        step = 0
        d_loss = [0]
        best_val_loss = 100000
        last_update_step = 0
        for epoch in range(self.num_epochs):
            print("="*15, "Epoch %d" % epoch, "="*15)
            batches_per_epoch = train_data_size // self.batch_size
            epoch_val_loss_sum = 0
            for batch_num in tqdm(range(batches_per_epoch)):
                batch_loss_sum = 0
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

                    batch_loss_sum += d_loss[0]

                batch_loss_avg = batch_loss_sum / self.c_itr

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
                
                # Training summaries
                d_loss_summary = self.makeSummary("d_loss", -batch_loss_avg)
                d_loss_writer.add_summary(d_loss_summary, step)
    
                g_feat_loss_summary = self.makeSummary("g_loss", feat_loss)
                g_feat_loss_writer.add_summary(g_feat_loss_summary, step)

                g_adv_loss_summary = self.makeSummary("g_loss", adv_loss)
                g_adv_loss_writer.add_summary(g_adv_loss_summary, step)
                
                # Evaluation summaries
                rand_indices = np.random.randint(0, test_data_size, size=self.batch_size)
                test_noisy_batch = self.test_input[rand_indices]
                test_reference_batch = self.test_labels[rand_indices]
                d_val_loss = self.critic_model.test_on_batch(
                    [test_reference_batch, test_noisy_batch], 
                    [real, fake, dummy]
                )

                d_val_loss_summary = self.makeSummary("d_loss", -d_val_loss[0])
                val_writer.add_summary(d_val_loss_summary, step)

                g_val_loss = self.generator_model.test_on_batch(
                    test_noisy_batch,
                    [test_reference_batch, real]
                )
                epoch_val_loss_sum += g_val_loss[1]

                step += 1

            epoch_val_loss_avg = epoch_val_loss_sum / batches_per_epoch
            if  epoch_val_loss_avg < best_val_loss:
                print("New best!")
                best_val_loss = epoch_val_loss_avg
                best_model = self.generator
                last_update_epoch = epoch

            g_val_loss_summary = self.makeSummary("g_loss", epoch_val_loss_avg)
            val_writer.add_summary(g_val_loss_summary, step)
    
            if epoch - last_update_epoch > 20:
                print("==== EARLY STOPPING =======")
                saveModel(best_model)
                return

            # Save the model at each epoch
            saveModel(best_model)

            val_psnr = psnr(epoch)
            val_psnr_summary = self.makeSummary("psnr", val_psnr)
            val_writer.add_summary(val_psnr_summary, step)
            print("FEAT_VAL_LOSS: %f" % epoch_val_loss_avg)
            print("PSNR: %f" % val_psnr)
            #print("D_LOSS: %f" % d_loss[0])
            #print("D_VAL_LOSS: %f" % batch_loss_avg)
            #print("G_ADV_LOSS: %f" % adv_loss)
                
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

    def makeSummary(self, tag, val):
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=val),]
        )
        return summary

    def makeSummaryWriter(self, directory, loss_function, name):
        writer = tf.summary.FileWriter(
            self.log_dir + "/{0}-{1}-{2}".format(
                self.timestamp, 
                self.g_lr,
                name
            ),
            max_queue=1,
            flush_secs=10
        )
        return writer
