import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from tensorflow.keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import array_to_img, img_to_array

import config
import data
from denoiser import Denoiser
from discriminator import Discriminator
import weighted_average

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

        # Set learning rates
        self.g_lr = kwargs.get("g_lr", 1e-3)
        self.d_lr = kwargs.get("d_lr", 1e-4)

        # Build and compile the Generator 
        self.generator = self.buildGenerator()
        self.compileGenerator()
        self.generator.summary()

        # Create our feature extractor for perceptual loss
        self.vgg = self.buildVGG()
        self.compileVGG()
        self.vgg.summary()

        # Build and compile the discriminator
        self.discriminator = self.buildDiscriminator()
        self.compileDiscriminator()
        self.discriminator.summary()

        # Build and compile the combined GAN model
        self.gan = self.buildGAN()
        self.compileGAN()
        self.gan.summary()

        # Initialise the instance noise parameter to 1
        self.discriminator_noise_parameter = 1

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
        
        vgg19.outputs = [vgg19.get_layer("block2_conv2").output] 
        
        img = tf.keras.layers.Input(
            shape=self.img_shape
        )

        # Extract image features
        img_features = vgg19(img)

        # Make the model and make it non-trainable
        model = tf.keras.models.Model(img, img_features)
        model.trainable = False

        return model

    def buildGenerator(self):

        # Helper function for convolutional layer
        def convLayer(c_input, num_filters):
            c_output = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[5, 5],
                use_bias=True,
                strides=[1, 1],
                padding="SAME",
                kernel_initializer="glorot_normal"
            )(c_input)
            return c_output

        ################################################

        # The generator takes a noisy image as input
        noisy_img = tf.keras.layers.Input(self.denoiser_input_shape, name="Generator_input")

        # 9 fully convolutional layers
        x = convLayer(noisy_img, 100)
        for _ in range(7):
            x = convLayer(x, 100)
            x = tf.keras.layers.ReLU()(x)

        # Final layer is not activated
        weights = convLayer(x, pow(self.kpcn_size, 2))

        return tf.keras.models.Model(noisy_img, weights, name="Generator")

    def buildDiscriminator(self):

        # Helper function for convolution layer
        def convBlock(c_input, num_filters, strides, bn=True):
            c_output = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                strides=strides,
                padding="SAME"
            )(c_input)
            
            output = tf.keras.layers.LeakyReLU(alpha=0.2)(c_output)
            
            if bn:
                output = tf.keras.layers.BatchNormalization(momentum=0.8)(output)
        
            return output

        ################################################

        img = tf.keras.layers.Input(shape=self.img_shape, name="Discriminator_input")
        
        x = convBlock(img, 64, strides=[1, 1], bn=False)
        x = convBlock(x, 64, strides=[2, 2])
        x = convBlock(x, 128, strides=[1, 1])
        x = convBlock(x, 128, strides=[2, 2])
        x = convBlock(x, 256, strides=[1, 1])
        x = convBlock(x, 256, strides=[2, 2])
        x = convBlock(x, 512, strides=[1, 1])
        x = convBlock(x, 512, strides=[2, 2])

        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        prob = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.models.Model(img, prob, name="Discriminator")

    def buildGAN(self):
        
        # Apply softmax function to normalise the weights
        def softmax(weights):
            exp = tf.math.exp(weights)
            weight_sum = tf.reduce_sum(exp, axis=3, keepdims=True)
            weights = tf.divide(exp, weight_sum)
            return weights

        # Prepare input data for weighted average function
        def processInputForKPCN(noisy_img):
            # Slice the noisy image out of the input
            noisy_img = noisy_img[:, :, :, 0:3]

            # Get the radius of the kernel
            kernel_radius = int(math.floor(self.kpcn_size / 2.0))

            # Pad the image on either side so that the kernel can reach all pixels
            paddings = tf.constant([[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])
            noisy_img = tf.pad(noisy_img, paddings, mode="SYMMETRIC")

            return noisy_img

        # Apply weighted average (KPCN)
        def applyKernel(noisy_img):
            import tensorflow as tf
            noisy_img = processInputForKPCN(noisy_img)
            def applyKernelLayer(weights):
                # Normalise the weights  (softmax)
                weights = softmax(weights)
                prediction = weighted_average.weighted_average(noisy_img, weights)
                return prediction
            return applyKernelLayer

        ################################################

        # Noisy images input
        noisy_img = tf.keras.layers.Input(self.denoiser_input_shape, name="GAN_Input")

        # Denoise the images and get their features from the vgg network
        weights = self.generator(noisy_img)

        denoised_img = tf.keras.layers.Lambda(applyKernel(noisy_img))(weights)

        denoised_features = self.vgg(
            self.preprocessVGG(denoised_img)
        )

        # Freeze the weights of the discriminator since we only want to train
        # the generator
        self.discriminator.trainable = False

        # Run the denoised image through the discriminator
        denoised_probability = self.discriminator(denoised_img)

        # Create sensible names for outputs in logs
        denoised_features = tf.keras.layers.Lambda(lambda x: x, name='Feature')(denoised_features)
        denoised_probability = tf.keras.layers.Lambda(lambda x: x, name='Adversarial')(denoised_probability)

        # Create model using binary cross entropy with reversed labels
        
        return tf.keras.models.Model(
            inputs=noisy_img,
            outputs=[denoised_probability, denoised_features, denoised_img], 
            name="GAN"
        )

    # ===== Functions to compile the models ===== #
    def compileVGG(self):
        adam = tf.keras.optimizers.Adam(
            lr=0.0001,
            beta_1=0.9
        )
        self.vgg.compile(
            loss="mse",
            optimizer=adam,
            metrics=["accuracy"]
        )

    def compileGenerator(self):
        adam = tf.keras.optimizers.Adam(
            lr=self.g_lr,
            beta_1=0.9,
            clipvalue=0.05
        )
        self.generator.compile(
            loss="mse",
            optimizer=adam,
            metrics=["mse"]
        )

    def compileDiscriminator(self):
        adam = tf.keras.optimizers.Adam(
            lr=self.d_lr,
            beta_1=0.9,
            clipvalue=0.05
        )
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=adam,
            metrics=["accuracy"]
        )

    def compileGAN(self):
        adam = tf.keras.optimizers.Adam(
            lr=self.g_lr,
            beta_1=0.9,
            clipvalue=0.05
        )
        self.gan.compile(
            loss=["binary_crossentropy", "mse", "mse"], # FINAL LOSS IS JUST SO WE CAN HAVE 3 OUTPUTS
            loss_weights=[1e-3, 0.006, 0], #Weighting from the paper
            optimizer=adam
        )

    # Convert from [0, 1] to [0, 255] for vgg
    def preprocessVGG(self, img):
        if isinstance(img, np.ndarray):
            return preprocess_input(img * 255.0)
        else:            
            return tf.keras.layers.Lambda(lambda x: preprocess_input(x * 255))(img)

    def setTensorBoardWriters(self):

        # ====== DISCRIMINATOR WRITERS ======
        self.discrim_real_writer = tf.summary.FileWriter(
            "../logs/" + self.model_dir + "/-{}/discrim_real".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.discrim_fake_writer = tf.summary.FileWriter(
            "../logs/" + self.model_dir + "/-{}/discrim_fake".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.discrim_loss_writer = tf.summary.FileWriter(
            "../logs/" + self.model_dir + "/-{}/discrim_loss".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        # ====== GENERATOR WRITERS =======
        self.gen_writer = tf.summary.FileWriter(
            "../logs/" + self.model_dir + "/-{}/gen".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.gen_f_writer = tf.summary.FileWriter(
            "../logs/" + self.model_dir + "/-{}/gen_feature".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.gen_f_val_writer = tf.summary.FileWriter(
            "../logs/" + self.model_dir + "/-{}/gen_val_feature".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.gen_a_writer = tf.summary.FileWriter(
            "../logs/" + self.model_dir + "/-{}/gen_adv".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        # ====== PSNR WRITERS ======
        self.psnr_val_writer = tf.summary.FileWriter(
            "../logs/" + self.model_dir + "/-{}/psnr_val".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

    def makeEpochSummariesAndWrite(self, val_psnr):#, val_f_loss):
        psnr_summary = tf.Summary(
            value=[tf.Summary.Value(tag="val_psnr", simple_value=val_psnr),]
        )

        #gen_feature_val_summary = tf.Summary(
        #    value=[tf.Summary.Value(tag="gen_loss", simple_value=val_f_loss),]
        #)

        self.psnr_val_writer.add_summary(psnr_summary, self.global_step)
        #self.gen_f_val_writer.add_summary(gen_feature_val_summary, self.global_step)
    
    
    def makeBatchSummariesAndWrite(                    
        self,
        discrim_pred_real,
        discrim_pred_fake,
        discrim_loss,
        avg_discrim_acc,
        gen_loss
    ):
        # Add the discriminator's loss to tensorboard summary
        discrim_pred_real_summary = tf.Summary(
            value=[tf.Summary.Value(tag="discrim_pred", simple_value=discrim_pred_real),]
        )
    
        discrim_pred_fake_summary = tf.Summary(
            value=[tf.Summary.Value(tag="discrim_pred", simple_value=discrim_pred_fake),]
        )

        # Add the adversarial loss to tensorboard sumamry
        discrim_loss_summary = tf.Summary(
            value=[tf.Summary.Value(tag="discrim_loss", simple_value=discrim_loss),]
        )

        avg_discrim_acc_summary = tf.Summary(
            value=[tf.Summary.Value(tag="discrim_acc", simple_value=avg_discrim_acc),]
        )
        
        # Add the gan's loss to tensorboard summary
        gen_summary = tf.Summary(
            value=[tf.Summary.Value(tag="gen_loss", simple_value=gen_loss[0]),]
        )
        gen_feature_summary = tf.Summary(
            value=[tf.Summary.Value(tag="gen_loss", simple_value=gen_loss[2] * 0.006),]
        )
        gen_adv_summary = tf.Summary(
            value=[tf.Summary.Value(tag="gen_loss", simple_value=gen_loss[1] * 1e-3),]
        )

        self.discrim_real_writer.add_summary(discrim_pred_real_summary, self.global_step)
        self.discrim_fake_writer.add_summary(discrim_pred_fake_summary, self.global_step)
        self.discrim_loss_writer.add_summary(discrim_loss_summary, self.global_step)
        self.discrim_loss_writer.add_summary(avg_discrim_acc_summary, self.global_step)
        self.gen_writer.add_summary(discrim_loss_summary, self.global_step)
        self.gen_f_writer.add_summary(gen_feature_summary, self.global_step)
        self.gen_a_writer.add_summary(gen_adv_summary, self.global_step)

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
        real_labels = np.ones(self.batch_size) * 1 #0.9# One sided label smoothing
        fake_labels = np.zeros(self.batch_size)
       
        # Add noise to discrim real input
        real_input_noise = np.random.normal(
            scale=self.discriminator_noise_parameter,
            size=self.img_shape
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
            size=self.img_shape
        )
        #fake_batch += fake_input_noise

        # Train discriminator of fake batch
        loss_fake = self.discriminator.train_on_batch(
            fake_batch,
            fake_labels
        )
        pred_real = loss_real[1]
        pred_fake = 1 - loss_fake[1]
        acc = 0.5 * np.add(loss_real[1], loss_fake[1])
        loss = 0.5 * np.add(loss_real[0], loss_fake[0])

        return (pred_real, pred_fake, acc, loss)

    def train(self):
        self.setTensorBoardWriters()

        train_data_size = np.array(self.train_data["noisy"]["diffuse"]).shape[0]
        test_data_size = np.array(self.test_data["noisy"]["diffuse"]).shape[0]

        print("  ========================================== ")
        print(" || Training on %d Patches                 ||" % train_data_size)
        print(" || Testing on %d Patches                  ||" % test_data_size)
        print("  ========================================== ")


        for epoch in range(self.num_epochs):
            print("="*15, "Epoch %d" % epoch, "="*15)
            epoch_discrim_acc = 0
            for batch_num in tqdm(range(train_data_size // self.batch_size)):

                # Get random numbers to select our batch
                rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)

                # Get a batch and denoise
                train_noisy_batch = self.train_input[rand_indices]
                train_reference_batch = self.train_labels[rand_indices]

                denoised_batch = self.gan.predict(train_noisy_batch)[2]

                (discrim_pred_real, discrim_pred_fake, discrim_acc, discrim_loss) = \
                    self.trainDiscriminator(
                        train_reference_batch,
                        denoised_batch
                    )

                # Get the features for the reference images
                reference_features = self.vgg.predict(
                    self.preprocessVGG(train_reference_batch)
                )

                # Train the generator
                gen_loss = self.gan.train_on_batch(
                    train_noisy_batch,
                    [np.ones(self.batch_size), reference_features, train_reference_batch]
                ) 

                # Write batch-based summaries to tensorboard
                self.makeBatchSummariesAndWrite(
                    discrim_pred_real,
                    discrim_pred_fake,
                    discrim_loss,
                    epoch_discrim_acc,
                    gen_loss
                )
            
                self.global_step += 1

            ref = train_reference_batch[0]
            denoised = denoised_batch[0]
            self.makeFigureAndSave(ref, denoised, epoch)

            # Calculate the feature loss and psnr on a test batch
            #val_f_loss, val_psnr = self.denoiser.eval(False)
            val_psnr = self.testPSNR()

            # Write epoch-based summaries to tensorboard
            self.makeEpochSummariesAndWrite(val_psnr)#, val_f_loss)
            
            # Decrease instance noise
            #if self.discriminator_noise_parameter > 0:
            #    self.discriminator_noise_parameter -= 0.01

            print("val_psnr: " + str(val_psnr))
            print("gen_feature_loss: " + str(gen_loss[2] * 0.006))
            print("gen_adversarial_loss: " + str(gen_loss[1] * 1e-3))

            self.gan.save("../models/" + self.model_dir + "-{}".format(self.timestamp))
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
        ref_subplot.plot(ref_img)
    
        denoised_subplot = plt.subplot(122)
        denoised_subplot.set_title("Denoised image")
        denoised_subplot.plot(ref_img)

        fig.savefig("../training_results/epoch:%d" % epoch)

    def testPSNR(self):
        reference_test_imgs = self.test_labels
        noisy_test_imgs = self.test_input
        denoised = self.gan.predict(noisy_test_imgs)[2]
        
        denoised = np.clip(denoised, 0, 1)
        reference_test_img = np.clip(denoised, 0, 1)

        with tf.Session(""):
            psnr = tf.image.psnr(denoised, reference_test_imgs, max_val=1.0)
            psnr = tf.where(tf.is_nan(psnr), tf.zeros_like(psnr), psnr)
            non_zeros = tf.count_nonzero(psnr)
            psnr = tf.divide(tf.reduce_sum(psnr), tf.cast(non_zeros, tf.float32)).eval()
        return psnr
        
