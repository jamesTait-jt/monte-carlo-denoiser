# Network models
from denoiser import Denoiser
from discriminator import Discriminator
from gan import GAN
import tungsten_data

# data helpers
import data

def main():

    patches = tungsten_data.getSampledPatches()

    train_data = patches["train"]
    test_data = patches["test"]

    # Set features to all
    feature_list = ["normal", "albedo", "depth"]
    
    # MAE - NO ALBDIV
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=100000, # Stupid number of epochs, early stopping will prevent reaching this
        early_stopping=True,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="mae",
        model_dir="../experiments/models/mae",
        log_dir="../experiments/logs/mae"
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    del denoiser

    # VGG - NO ALBDIV
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-3,
        num_epochs=100000,
        early_stopping=True,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="vgg22",
        model_dir="../experiments/models/vgg",
        log_dir="../experiments/logs/vgg",
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    del denoiser

    # WGAN-GP - NO ALBDIV
    gan = GAN(
        train_data, 
        test_data, 
        num_epochs=100000,
        kernel_predict=True,
        batch_size=64,
        g_layers=3,
        g_loss="vgg",
        loss_weights=[0.1, 1.0],
        g_lr=1e-3,
        g_kernel_size=[3, 3],
        g_beta1=0.5,
        g_beta2=0.9,
        c_lr=3e-4,
        c_itr=1,
        g_bn=True,
        model_dir="../experiments/models/wgan-gp",
        log_dir="../experiments/logs/wgan-gp",
    )
    gan.trainWGAN_GP()

if __name__ == "__main__":
    main()
