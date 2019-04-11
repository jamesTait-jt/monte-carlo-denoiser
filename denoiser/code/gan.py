import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm
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

        self.num_epochs = kwargs.get("num_epochs", 10)

        # The adam optimiser is used, this block defines its parameters
        self.adam_lr = kwargs.get("adam_lr", 1e-3)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.999)
        self.adam_lr_decay = kwargs.get("adam_lr_decay", 0.0)

        self.adam = tf.keras.optimizers.Adam(
            lr=self.adam_lr,
            beta_1=self.adam_beta1,
            beta_2=self.adam_beta2,
            decay=self.adam_lr_decay,
            clipnorm=1,
            clipvalue=0.05
        )

        self.batch_seed = kwargs.get("batch_seed", 1357)
        #np.random.seed(self.batch_seed)

        self.batch_size = kwargs.get("batch_size", 64)

        # Initialise the instance noise parameter to 1
        self.discriminator_noise_parameter = 1

        # Set the number of update interations to 0
        self.global_step = 0

        # Set the model's timestamp
        self.timestamp = time()

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
            
        #predicted_img = tf.keras.layers.ReLU()(predicted_img)
        #predicted_img = tf.keras.layers.Activation("tanh")(predicted_img)

        discrim_output = self.discriminator.model(predicted_img)
        self.model = tf.keras.models.Model(inputs=gan_input, outputs=[predicted_img, discrim_output])
        
        if self.denoiser.kernel_predict:
            loss = self.denoiser.VGG19FeatureLoss
        else:
            loss = "mean_squared_error"
            metrics = [self.psnr]

        self.model.compile(
            loss=[loss, "binary_crossentropy"],
            #loss_weights=[1.0, 1e-3],
            loss_weights=[1, 0.1],
            metrics=["accuracy"],
            #loss_weights=[1.0, 0],
            optimizer=self.adam
        )
        
    def setTensorBoardWriters(self):

        # ====== DISCRIMINATOR WRITERS ======
        self.discrim_real_writer = tf.summary.FileWriter(
            "../logs/gan-{}/discrim_real".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.discrim_fake_writer = tf.summary.FileWriter(
            "../logs/gan-{}/discrim_fake".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.discrim_loss_writer = tf.summary.FileWriter(
            "../logs/gan-{}/discrim_loss".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        # ====== GENERATOR WRITERS =======
        self.gen_writer = tf.summary.FileWriter(
            "../logs/gan-{}/gen".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.gen_f_writer = tf.summary.FileWriter(
            "../logs/gan-{}/gen_feature".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.gen_f_val_writer = tf.summary.FileWriter(
            "../logs/gan-{}/gen_val_feature".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        self.gen_a_writer = tf.summary.FileWriter(
            "../logs/gan-{}/gen_adv".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

        # ====== PSNR WRITERS ======
        self.psnr_val_writer = tf.summary.FileWriter(
            "../logs/gan-{}/psnr_val".format(self.timestamp),
            max_queue=1,
            flush_secs=10
        )

    def makeEpochSummariesAndWrite(self, val_psnr, val_f_loss):
        psnr_summary = tf.Summary(
            value=[tf.Summary.Value(tag="val_psnr", simple_value=val_psnr),]
        )

        gen_feature_val_summary = tf.Summary(
            value=[tf.Summary.Value(tag="gen_loss", simple_value=val_f_loss),]
        )

        self.psnr_val_writer.add_summary(psnr_summary, self.global_step)
        self.gen_f_val_writer.add_summary(gen_feature_val_summary, self.global_step)
    
    
    def makeBatchSummariesAndWrite(                    
        self,
        discrim_pred_real,
        discrim_pred_fake,
        discrim_loss,
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
        
        # Add the gan's loss to tensorboard summary
        gen_summary = tf.Summary(
            value=[tf.Summary.Value(tag="gen_loss", simple_value=gen_loss[0]),]
        )
        gen_feature_summary = tf.Summary(
            value=[tf.Summary.Value(tag="gen_loss", simple_value=gen_loss[1]),]
        )
        gen_adv_summary = tf.Summary(
            value=[tf.Summary.Value(tag="gen_loss", simple_value=gen_loss[2] * 0.1),]
        )

        self.discrim_real_writer.add_summary(discrim_pred_real_summary, self.global_step)
        self.discrim_fake_writer.add_summary(discrim_pred_fake_summary, self.global_step)
        self.discrim_loss_writer.add_summary(discrim_loss_summary, self.global_step)
        self.gen_writer.add_summary(discrim_loss_summary, self.global_step)
        self.gen_f_writer.add_summary(gen_feature_summary, self.global_step)
        self.gen_a_writer.add_summary(gen_adv_summary, self.global_step)

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
            num_layers=7,
            loss="vgg22",
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
        real_input_noise = np.random.normal(scale=self.discriminator_noise_parameter)
        real_batch += real_input_noise

        # Train discriminator on real batch
        loss_real = self.discriminator.model.train_on_batch(
            real_batch, 
            real_labels
        )

        # Add noise to discrim fake batch
        fake_input_noise = np.random.normal(scale=self.discriminator_noise_parameter)
        fake_batch += fake_input_noise

        # Train discriminator of fake batch
        loss_fake = self.discriminator.model.train_on_batch(
            fake_batch,
            fake_labels
        )
        pred_real = loss_real[1]
        pred_fake = 1 - loss_fake[1]
        acc = 0.5 * np.add(loss_real[1], loss_fake[1])
        loss = 0.5 * np.add(loss_real[0], loss_fake[0])

        return (pred_real, pred_fake, acc, loss)

    def trainCombinedModel(self, train_data_size):
        rand_indices = np.random.randint(0, train_data_size, size=self.batch_size)
        noisy_batch = self.denoiser.train_input[rand_indices]
        reference_batch = self.denoiser.train_labels[rand_indices]

        # Create labels for the gan
        labels = np.ones(self.batch_size)
        loss = self.model.train_on_batch(noisy_batch, [reference_batch, labels])
        
        return loss

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
                train_noisy_batch = self.denoiser.train_input[rand_indices]
                denoised_batch = self.model.predict(train_noisy_batch)[0]

                train_reference_batch = self.denoiser.train_labels[rand_indices]

                #print("ref: " , np.amax(train_reference_batch))
                #print("ref: " , np.amin(train_reference_batch))
                #print("denoised: " , np.amax(denoised_batch))
                #print("denoised: " , np.amin(denoised_batch))

                # Unfreeze the discriminator so that we can train it
                self.discriminator.model.trainable = True

                # If we just started a batch - train discrim
                if batch_num <= (train_data_size // self.batch_size) // 3:
                    (discrim_pred_real, discrim_pred_fake, discrim_acc, discrim_loss) = \
                        self.trainDiscriminator(
                            train_reference_batch,
                            denoised_batch
                        )
                    epoch_discrim_acc += discrim_acc
                    epoch_discrim_acc /= (batch_num + 1)

                # If discrim is still bad, train it
                elif epoch_discrim_acc <= 0.5:
                    (discrim_pred_real, discrim_pred_fake, discrim_acc, discrim_loss) = \
                        self.trainDiscriminator(
                            train_reference_batch,
                            denoised_batch
                        )
                    epoch_discrim_acc += discrim_acc
                    epoch_discrim_acc /= (batch_num + 1)
                # Else skip the rest of the batch
                else:
                    print(epoch_discrim_acc)
                    print("Skip training discrim")
                    epoch_discrim_acc = 1


                # Freeze the weights so the discriminator isn't trained with the denoiser in the gan network
                self.discriminator.model.trainable = False
                
                generator_itrs = 1
                for _ in range(generator_itrs):
                    # Train the combined model (generator)
                    gen_loss = self.trainCombinedModel(train_data_size)
                    #discrim_acc = 1 - gen_loss[4]

                # Write batch-based summaries to tensorboard
                self.makeBatchSummariesAndWrite(
                    discrim_pred_real,
                    discrim_pred_fake,
                    discrim_loss,
                    gen_loss
                )
            
                self.global_step += 1

            # Calculate the feature loss and psnr on a test batch
            val_f_loss, val_psnr = self.denoiser.eval(False)

            # Write epoch-based summaries to tensorboard
            self.makeEpochSummariesAndWrite(val_psnr, val_f_loss)
            
            # Decrease instance noise
            #if self.discriminator_noise_parameter > 0:
            #    self.discriminator_noise_parameter -= 0.01

            print("val_psnr: " + str(val_psnr))
            print("gen_feature_loss: " + str(gen_loss[1]))
            print("gen_adversarial_loss: " + str(gen_loss[2] * 0.1))
            print("gen_val_f_loss: " + str(val_f_loss))

            self.denoiser.model.save("../models/gan-{}".format(self.timestamp))
            #self.model.save("../models/gan-{}".format(self.timestamp))
            if (epoch % 50) == 0:
                self.denoiser.model.save(self.denoiser.model_dir + "epoch:" + str(epoch))


    def eval(self):
        reference_test_images = np.array(self.test_data["reference"]["diffuse"])
        reference_test_labels = np.ones(len(reference_test_images))
        if config.ALBEDO_DIVIDE:
            reference_test_images = np.array(self.test_data["reference"]["albedo_divided"])

        score = self.model.evaluate(self.denoiser.test_input, [reference_test_images, reference_test_labels], verbose=0)
        #print(" ")
        #print(" ===== GAN EVALUATION ===== ")
        print("gan_val_loss: " + str(score[0]))
        #print(" ")
        return score[0]
