import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm

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
            decay=self.adam_lr_decay,
            clipnorm=1,
            clipvalue=0.05,
            #amsgrad=True
        )

        self.batch_seed = kwargs.get("batch_seed", 1357)
        np.random.seed(self.batch_seed)

        self.batch_size = kwargs.get("batch_size", 64)

        self.buildDenoiser()
        self.buildDiscriminator()
        self.buildNetwork()
        
    def psnr(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    # Lambda layer to apply the kernel inside the network
    def applyKernel(self, noisy_img):
        import tensorflow as tf
        noisy_img = self.denoiser.processImgForKernelPrediction(noisy_img)
        def applyKernelLayer(weights):
            weights = self.denoiser.processWeightsForKernelPrediction(weights)
            denoiser_output = weighted_average.weighted_average(noisy_img, weights)
            return denoiser_output
        return applyKernelLayer

    def buildNetwork(self):
        self.discriminator.model.trainable = False
        gan_input = tf.keras.layers.Input((config.PATCH_HEIGHT, config.PATCH_WIDTH, self.denoiser.input_channels))
        denoiser_output = self.denoiser.model(gan_input)

        if self.denoiser.kernel_predict:
            predicted_img = tf.keras.layers.Lambda(self.applyKernel(gan_input))(denoiser_output)

        discrim_output = self.discriminator.model(predicted_img)
        self.model = tf.keras.models.Model(inputs=gan_input, outputs=[denoiser_output, discrim_output])
        
        if self.denoiser.kernel_predict:
            loss = self.denoiser.MAVKernelPredict(gan_input)
            metrics = [self.denoiser.kernelPredictPSNR(gan_input)]
        else:
            loss = "mean_squared_error"
            metrics = [self.psnr]

        self.model.compile(
            loss=[loss, "binary_crossentropy"],
            loss_weights=[1.0, 1e-2],
            metrics=["accuracy"],
            #loss_weights=[1.0, 0],
            optimizer=self.adam
        )
        

    def preTrainMAVDenoiser(self):
        feature_list = ["normal", "albedo", "depth"]
        mav_denoiser = Denoiser(
            self.train_data,
            self.test_data,
            mse_epochs=20,
            vgg_epochs=0,
            kernel_predict=True,
            feature_list=feature_list
        )
        if os.path.isfile("../models/mav_denoiser"):
            print("Loading in initial mav denoiser...")
            mav_denoiser.model = tf.keras.models.load_model(
                "../models/mav_denoiser",
                custom_objects={"psnr" : mav_denoiser.psnr}
            )
            mav_denoiser.eval(True)
        else:
            print("No mav_denoiser found - training now...")
            mav_denoiser.buildNetwork()
            mav_denoiser.train()
            mav_denoiser.eval(True)
            mav_denoiser.model.save("../models/mav_denoiser")

        self.mav_denoiser = mav_denoiser
        self.denoiser = mav_denoiser

    def buildDenoiser(self):
        feature_list = ["normal", "albedo", "depth"]
        denoiser = Denoiser(
            self.train_data,
            self.test_data,
            kernel_predict=True,
            batch_size=self.batch_size,
            feature_list=feature_list
        )
        denoiser.buildNetwork()
        self.denoiser = denoiser


    def buildDiscriminator(self):
        discriminator = Discriminator(
            self.train_data,
            self.test_data,
            batch_size=self.batch_size
        )
        discriminator.buildNetwork()
        discriminator.compile()
        self.discriminator = discriminator


    # Train the discriminator on the output of the mean absolute value denoiser
    def preTrainDiscriminator(self, mav_train_pred, mav_test_pred):
        discriminator = Discriminator(
            self.train_data,
            self.test_data,
            mav_test_pred,
            num_epochs=1000,
        )

        if os.path.isfile("../models/discriminator"):
            print("Loading in discriminator...")
            discriminator.model = tf.keras.models.load_model("../models/discriminator") 
            discriminator.eval()
        else:
            print("No discriminator found - training now...")
            discriminator.buildNetwork()
            #discriminator.train()
            discriminator.eval()
            discriminator.model.save("../models/discriminator")

        self.discriminator = discriminator

    def denoiserPredict(self):
        print("Evaluating denoiser on train data...")
        train_pred = self.denoiser.predict(self.train_data)
        print("Evaluating denoiser on test data...")
        test_pred = self.denoiser.predict(self.test_data)
        return train_pred, test_pred

    def train(self):
        #self.denoiser.discriminator = self.discriminator
        
        now = time()
        gan_writer = tf.summary.FileWriter(
            "../logs/gan-{}".format(now),
            max_queue=1,
            flush_secs=10
        )
        gan_val_writer = tf.summary.FileWriter(
            "../logs/gan_val-{}".format(now), 
            max_queue=1,
            flush_secs=10
        )
        adversarial_writer = tf.summary.FileWriter(
            "../logs/adversarial-{}".format(now), 
            max_queue=1,
            flush_secs=10
        )

        self.denoiser.vgg_mode = 54

        train_data_size = np.array(self.train_data["noisy"]["diffuse"]).shape[0]
        test_data_size = np.array(self.test_data["noisy"]["diffuse"]).shape[0]

        print("  ========================================== ")
        print(" || Training on %d Patches                 ||" % train_data_size)
        print(" || Testing on %d Patches                  ||" % test_data_size)
        print("  ========================================== ")

        global_step = 0
        for epoch in range(self.num_epochs):
            print("="*15, "Epoch %d" % epoch, "="*15)
            for _ in tqdm(range(train_data_size // self.batch_size)):

                # Get random numbers to select our batch
                rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)

                # Get a batch and denoise
                train_noisy_batch = self.denoiser.train_input[rand_indices]
                train_reference_batch = np.array(
                    self.denoiser.train_data["reference"]["diffuse"]
                )[rand_indices]

                denoised_images = self.denoiser.model.predict(train_noisy_batch)

                # Create labels for the discriminator
                train_reference_labels = np.ones(self.batch_size) * 0.9 # One sided label smoothing
                train_noisy_labels = np.zeros(self.batch_size)

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
                train_reference_batch = np.array(
                    self.denoiser.train_data["reference"]["diffuse"]
                )[rand_indices]

                # Create labels for the gan
                gan_labels = np.ones(self.batch_size)

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

                gan_acc_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="gan_acc", simple_value=gan_loss[1]),]
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

            print("discriminator_loss_real :" + str(discrim_loss_real[0]))
            print("discriminator_loss_fake :" + str(discrim_loss_fake[0]))
            print("gan_loss :" + str(gan_loss[0]))
            print("psnr :" + str(psnr))

            self.denoiser.model.save("../models/gan-{}".format(now))
            if (epoch % 50) == 0:
                self.denoiser.model.save(self.denoiser.model_dir + "epoch:" + str(epoch))

            # Drop the LR after 200 epochs
            if (epoch == 100):
                self.adam_lr = 1e-5
                self.adam = tf.keras.optimizers.Adam(
                    lr=self.adam_lr,
                    beta_1=self.adam_beta1,
                    beta_2=self.adam_beta2,
                    decay=self.adam_lr_decay,
                    clipnorm=1,
                    #amsgrad=True
                )
                # Need to recompile with the new LR
                self.buildModel()


    def eval(self):
        reference_test_images = np.array(self.test_data["reference"]["diffuse"])
        reference_test_labels = np.ones(len(reference_test_images))
        score = self.model.evaluate(self.denoiser.test_input, [reference_test_images, reference_test_labels], verbose=0)
        #print(" ")
        #print(" ===== GAN EVALUATION ===== ")
        print("gan_val_loss: " + str(score[0]))
        #print(" ")
        return score[0]
