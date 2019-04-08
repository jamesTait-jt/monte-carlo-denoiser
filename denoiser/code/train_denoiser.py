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
        num_epochs=200,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=4
    )
    denoiser.buildNetwork()
    #denoiser.train()

    gan = GAN(
        train_data,
        test_data,
        num_epochs=1000,
        batch_size=16
    )
    gan.train()

if __name__ == "__main__":
    main()
