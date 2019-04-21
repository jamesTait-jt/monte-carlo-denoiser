# Network models
from denoiser import Denoiser
from discriminator import Discriminator
from gan import GAN
import tungsten_data

# data helpers
import data
import config

def mae(train_data, test_data):
    """Train the MAE denoiser with no albedo divide and no batch norm. Use all
    features."""
    config.ALBEDO_DIVIDE = False
    feature_list = ["normal", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=100000, # Stupid number of epochs, early stopping will prevent reaching this
        early_stopping=True,
        kernel_predict=True,
        bn=False,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="mae",
        model_dir="../experiments/models/mae",
        log_dir="../experiments/logs/mae"
    )
    denoiser.buildNetwork()
    denoiser.train()

def mae_bn(train_data, test_data):
    """Train the MAE denoiser with no albedo divide and batch norm on. Use all
    features."""
    config.ALBEDO_DIVIDE = False
    feature_list = ["normal", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-3,
        num_epochs=100000, # Stupid number of epochs, early stopping will prevent reaching this
        early_stopping=True,
        kernel_predict=True,
        bn=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="mae",
        model_dir="../experiments/models/mae_bn",
        log_dir="../experiments/logs/mae_bn"
    )
    denoiser.buildNetwork()
    denoiser.train()

def mae_albdiv(train_data, test_data):
    """Train the MAE denoiser with albedo divide and no batch norm. Use all
    features."""
    config.ALBEDO_DIVIDE = True
    feature_list = ["normal", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=100000, # Stupid number of epochs, early stopping will prevent reaching this
        early_stopping=True,
        kernel_predict=True,
        bn=False,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="mae",
        model_dir="../experiments/models/mae_albdiv",
        log_dir="../experiments/logs/mae_albdiv"
    )
    denoiser.buildNetwork()
    denoiser.train()

def mae_albdiv_bn(train_data, test_data):
    """Train the MAE denoiser with albedo divide and batch norm on. Use all
    features."""
    config.ALBEDO_DIVIDE = True
    feature_list = ["normal", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=100000, # Stupid number of epochs, early stopping will prevent reaching this
        early_stopping=True,
        kernel_predict=True,
        bn=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="mae",
        model_dir="../experiments/models/mae_albdiv_bn",
        log_dir="../experiments/logs/mae_albdiv_bn"
    )
    denoiser.buildNetwork()
    denoiser.train()

def vgg(train_data, test_data):
    """Train the vgg denoiser with no albedo divide and no batch norm. Use all
    features"""
    config.ALBEDO_DIVIDE = False
    feature_list = ["normal", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=100000,
        early_stopping=True,
        kernel_predict=True,
        bn=False,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="vgg22",
        model_dir="../experiments/models/vgg",
        log_dir="../experiments/logs/vgg",
    )
    denoiser.buildNetwork()
    denoiser.train()


def vgg_bn(train_data, test_data):
    """Train the vgg denoiser with no albedo divide and batch norm on. Use all
    features"""
    config.ALBEDO_DIVIDE = False
    feature_list = ["normal", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-3,
        num_epochs=100000,
        early_stopping=True,
        kernel_predict=True,
        bn=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="vgg22",
        model_dir="../experiments/models/vgg_bn",
        log_dir="../experiments/logs/vgg_bn",
    )
    denoiser.buildNetwork()
    denoiser.train()


def vgg(train_data, test_data):
    """Train the vgg denoiser with no albedo divide and no batch norm. Use all
    features"""
    config.ALBEDO_DIVIDE = False
    feature_list = ["normal", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=100000,
        early_stopping=True,
        kernel_predict=True,
        bn=False,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="vgg22",
        model_dir="../experiments/models/vgg",
        log_dir="../experiments/logs/vgg",
    )
    denoiser.buildNetwork()
    denoiser.train()


def main():

    patches = tungsten_data.getSampledPatches()

    train_data = patches["train"]
    test_data = patches["test"]

    # ==== NO ALBDIV, NO BATCHNORM, ALL FEATURES ==== #
    #mae(train_data, test_data)
    #vgg(train_data, test_data)

    # === NO ALBDIV, BATCHNORM ON, ALL FEATURES ==== #
    #mae_bn(train_data, test_data)
    #vgg_bn(train_data, test_data)

    # === ALBDIV ON, NO BATCHNORM, ALL FEATURES ==== #
    mae_albdiv(train_data, test_data)
    #vgg_albdiv(train_data, test_data)

    # === ALBDIV ON, BATCHNORM ON, ALL FEATURES ==== #
    mae_albdiv_bn(train_data, test_data)
    #vgg_albdiv_bn(train_data, test_data)


    """
    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-5,
        num_epochs=100000, # Stupid number of epochs, early stopping will prevent reaching this
        early_stopping=True,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="ssim",
        model_dir="../experiments/models/ssim",
        log_dir="../experiments/logs/ssim"
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
        c_lr=1e-4,
        c_itr=5,
        g_bn=True,
        model_dir="../experiments/models/wgan-gp",
        log_dir="../experiments/logs/wgan-gp",
    )
    gan.trainWGAN_GP()

    # SET ALBEDO DIVIDE ON
    config.ALBEDO_DIVIDE = True

    # MAE - ALBDIV
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
        model_dir="../experiments/models/mae-albdiv",
        log_dir="../experiments/logs/mae-albdiv"
    )
    denoiser.buildNetwork()
    denoiser.train()
    del denoiser

    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-5,
        num_epochs=100000, # Stupid number of epochs, early stopping will prevent reaching this
        early_stopping=True,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=64,
        loss="ssim",
        model_dir="../experiments/models/ssim",
        log_dir="../experiments/logs/ssim"
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    del denoiser

    # VGG - ALBDIV
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
        model_dir="../experiments/models/vgg-albdiv",
        log_dir="../experiments/logs/vgg-albdiv",
    )
    denoiser.buildNetwork()
    denoiser.train()
    del denoiser

    # WGAN-GP - ALBDIV
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
        c_lr=1e-4,
        c_itr=5,
        g_bn=True,
        model_dir="../experiments/models/wgan-gp-albdiv",
        log_dir="../experiments/logs/wgan-gp-albdiv",
    )
    gan.trainWGAN_GP()
    """

if __name__ == "__main__":
    main()
