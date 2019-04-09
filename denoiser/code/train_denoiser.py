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
        adam_lr=1e-4,
        num_epochs=200000, # Stupid number of epochs, early stopping will prevent reaching this
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=32,
        loss="mae",
        model_dir="mae_0.0001_allfeatures"
    )
    denoiser.buildNetwork()
    #denoiser.train()
    del denoiser

    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=200000,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=32,
        loss="vgg22",
        model_dir="vgg22_0.0001_allfeatures"
    )
    denoiser.buildNetwork()
    denoiser.compile()
    del denoiser

    denoiser = Denoiser(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=200000,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=32,
        loss="vgg54",
        model_dir="vgg54_0.0001_allfeatures"
    )
    denoiser.buildNetwork()
    denoiser.compile()

    discriminator = Discriminator(
        train_data,
        test_data,
        adam_lr=1e-4,
        num_epochs=200,
        batch_size=32
    )
    discriminator.buildNetwork()
    discriminator.compile()
    #discriminator.train()

    gan = GAN(
        train_data, 
        test_data, 
        num_epochs=1000,
        batch_size=32
    )
    #gan.train()

if __name__ == "__main__":
    main()
