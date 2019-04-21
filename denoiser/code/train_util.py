import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

from keras.preprocessing.image import array_to_img, img_to_array

import weighted_average

def meanWithoutNanOrInf(tensor):
    """Replaces all of the nans and infs in a tensor with zero, then calculates
    the mean of the non-zero elements."""
    tensor = tf.where(tf.is_inf(tensor), tf.zeros_like(tensor), tensor)
    tensor = tf.where(tf.is_nan(tensor), tf.zeros_like(tensor), tensor)

    # Count how many non zeros we have (corresponding to how many wer
    # not nan or inf)
    non_zeros = tf.count_nonzero(tensor)

    # Sum up the tensor
    tensor_sum = K.sum(tensor)
    
    # Divide the batch sum by the number of non zero (nan or inf)
    # elements
    mean = tensor_sum / K.cast(non_zeros, "float32") #tf.cast(non_zeros, tf.float32))

    return mean

def processImgForKernelPrediction(noisy_img, kpcn_size):
    """Slice the image out of the network input and pad it so kernel can cover
    every pixel"""

    # Slice the noisy image out of the input
    noisy_img = noisy_img[:, :, :, 0:3]

    # Get the radius of the kernel
    kernel_radius = int(math.floor(kpcn_size / 2.0))

    # Pad the image on either side so that the kernel can reach all pixels
    paddings = tf.constant([[0, 0], [kernel_radius, kernel_radius], [kernel_radius, kernel_radius], [0, 0]])
    noisy_img = tf.pad(noisy_img, paddings, mode="SYMMETRIC")

    return noisy_img

def softmax(weights):
    """Apply the softmax function to the weights. SUbtract a constant to avoid
    integer overflow.
        softmax = exp(x) / sum(exp(x))
    """
    weightsum = tf.reduce_max(weights, axis=3, keepdims=True)
    weights = weights - weightsum
    exp = tf.math.exp(weights)
    weight_sum = tf.reduce_sum(exp, axis=3, keepdims=True)
    weights = tf.divide(exp, weight_sum)
    return weights

def applyKernel(noisy_img, weights, kpcn_size):
    """Preprocess the weights and image and apply the kernel."""
    weights = softmax(weights)
    noisy_img = processImgForKernelPrediction(noisy_img, kpcn_size)
    prediction = weighted_average.weighted_average(noisy_img, weights)
    return prediction

def applyKernelLambda(noisy_img, kpcn_size):
    """Preprocess weights and image and apply kernel as a lambda layer."""
    import tensorflow as tf
    def layer(weights):
        return applyKernel(noisy_img, weights, kpcn_size)
    return layer

def makeSummary(tag, val):
    """Make a tensorboard summary out of a float value."""
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=val),]
    )
    return summary


def makeSummaryWriter(timestamp, lr, name, log_dir):
    """Create a summary writer to write logs to tensorboard."""
    writer = tf.summary.FileWriter(
        log_dir + "/{0}-{1}-{2}".format(
            timestamp, 
            lr,
            name
        ),
        max_queue=1,
        flush_secs=10
    )
    return writer

def makeFigureAndSave(ref, denoised, epoch):
    """Make a plt plot of the reference image next to the denoised image and
    saves it."""
    ref = ref.clip(0, 1)
    denoised = denoised.clip(0, 1)

    ref_img = array_to_img(ref)
    denoised_img = array_to_img(denoised)

    fig = plt.figure()

    ref_subplot = plt.subplot(121)
    ref_subplot.set_title("Reference image")
    ref_subplot.imshow(ref_img)

    denoised_subplot = plt.subplot(122)
    denoised_subplot.set_title("Denoised image")
    denoised_subplot.imshow(denoised_img)

    fig.savefig("../training_results/epoch:%d" % epoch)

