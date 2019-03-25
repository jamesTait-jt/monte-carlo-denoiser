# Network models
from denoiser import Denoiser
from discriminator import Discriminator
from gan import GAN

# data helpers
import data

def main():

    seed = 1234
    patches = data.makePatches(seed)

    train_data = patches["train"]
    test_data = patches["test"]

    feature_list = ["sn", "albedo", "depth"]
    denoiser = Denoiser(
        train_data,
        test_data,
        num_epochs=100,
        kernel_predict=True,
        feature_list=feature_list
    )
    denoiser.buildNetwork()
    #denoiser.train()

    gan = GAN(train_data, test_data, num_epochs=1000)
    gan.train()

if __name__ == "__main__":
    main()
