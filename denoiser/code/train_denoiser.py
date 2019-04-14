# Network models
from denoiser import Denoiser
from discriminator import Discriminator
from gan import GAN
import tungsten_data

# data helpers
import data

def main():

    seed = 1234
    #patches = data.makePatches(seed)
    #patches = tungsten_data.getPatches()
    patches = tungsten_data.getSampledPatches()

    train_data = patches["train"]
    test_data = patches["test"]

    feature_list = ["normal", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        early_stopping=True,
        adam_lr=1e-3,
        num_epochs=200000, # Stupid number of epochs, early stopping will prevent reaching this
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=32,
        loss="mae",
        model_dir="mae_0.001_0.00001_allfeatures_1"
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    del denoiser
    
    denoiser = Denoiser(
        train_data,
        test_data,
        early_stopping=True,
        adam_lr=1e-5,
        num_epochs=200000, # Stupid number of epochs, early stopping will prevent reaching this
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=32,
        loss="mae",
        model_dir="mae_0.00001_allfeatures_1"
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    #del denoiser

    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-3,
        num_epochs=200000,
        early_stopping=True,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=32,
        loss="vgg22",
        model_dir="vgg22_0.001_allfeatures"
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    #del denoiser

    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-3,
        num_epochs=200000,
        early_stopping=True,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=32,
        loss="combination",
        model_dir="combination_0.001_allfeatures"
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    #del denoiser

    discriminator = Discriminator(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=200,
        batch_size=32
    )
    #discriminator.buildNetwork()
    #discriminator.compile()
    #discriminator.train()

    gan = GAN(
        train_data, 
        test_data, 
        num_epochs=1000,
        batch_size=64,
        model_dir="gan_0vgg22_1adv_0.001lr_0.9b1_0.999b2_64bs_allfeatures"
    )
    gan.trainWGAN()

if __name__ == "__main__":
    main()
