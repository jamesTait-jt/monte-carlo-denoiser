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
        early_stopping=True,
        kernel_predict=True,
        feature_list=feature_list,
        num_layers=7,
        batch_size=32,
        loss="mae",
        model_dir="MAE-0.0001-allfeatures-albedodiv"
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    del denoiser

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
        model_dir="vgg-0.001-allfeatures-albedodiv"
    )
    #denoiser.buildNetwork()
    #denoiser.train()
    del denoiser

    gan = GAN(
        train_data, 
        test_data, 
        num_epochs=1000,
        kernel_predict=True,
        batch_size=32,
        g_lr=1e-4,
        g_beta1=0.5,
        g_beta2=0.9,
        c_lr=1e-4,
        c_itr=10
    )
    gan.trainWGAN_GP()
    #gan.toyGenerator()

if __name__ == "__main__":
    main()
