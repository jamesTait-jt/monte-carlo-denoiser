import numpy as np
import tensorflow as tf

import eval_util
import tungsten_data

def getSinglePSNR(model, images, index):
    """Get the PSNR values for a full image of index (index) given a model."""
    denoised = eval_util.denoiseFullTestImg(model, images, index) 
    reference = images["test"]["reference"]["diffuse"][index]
    psnr = eval_util.psnr(denoised, reference, max_val=1.0)
    print(psnr)



model_paths = [
    "../experiments/models/mae_albdiv",
    "../experiments/models/vgg_albdiv",
    "../experiments/models/wgan-gp_albdiv"
]

models = eval_util.loadModels(model_paths)

# Load in the patches to evaluate psnr
#patches_dict = tungsten_data.loadPkl(tungsten_data.patches_pkl_path)

# Load in full test images to evaluate psnr
images = tungsten_data.loadPkl(tungsten_data.full_pkl_path)

getSinglePSNR(models[0], images, 0)

#patches, noisy_imgs = eval_util.getPatchesAsInput(patches_dict)


#inpt = np.expand_dims(patches[0], axis=0)
#noisy_img = np.expand_dims(noisy_imgs[0], axis=0)
#inpt = patches[0:20]
#noisy_imgs = noisy_imgs[0:20]

#weights = models[0].predict(inpt)

#denoised_patches = []
#for i in range(len(inpt)):
#    noisy_img = np.expand_dims(noisy_imgs[i], axis=0)
#    img_weights = np.expand_dims(weights[i], axis=0)
#    denoised = eval_util.applyKernel(noisy_img, img_weights)
#    denoised_patches.append(eval_util.albedoMultiply(patches_dict, denoised, i))

#reference = patches_dict["test"]["reference"]["diffuse"][0:20]

#denoised_patches = np.array(denoised_patches)
#reference = np.array(reference)


#denoised_patches_tensor = tf.placeholder(

#psnr = tf.reduce_mean(
#    tf.image.psnr(denoised_patches, reference, max_val=1.0)
#)

