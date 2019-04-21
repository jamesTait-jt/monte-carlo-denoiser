import tensorflow as tf
import keras
import keras.backend as K

import train_util

def kpcnMAE(noisy_img, kpcn_size):
    """Apply the kernel and calculate the MAE (mean absolute error) pixel-wise
    loss between the prediction and the ground truth."""
    def loss(y_true, y_pred):
        prediction = train_util.applyKernel(noisy_img, y_pred, kpcn_size)

        mae = keras.losses.mean_absolute_error(y_true, prediction)

        # Remove all NaN
        mae = train_util.meanWithoutNanOrInf(mae)
        mae = mae * 100

        return mae
    return loss

def kpcnMSE(noisy_img, kpcn_size):
    """Apply the kernel and calculate the MSE (mean squared error) pixel-wise
    loss between the prediction and the ground truth."""
    def loss(y_true, y_pred):
        prediction = train_util.applyKernel(noisy_img, y_pred, kpcn_size)
        mse = keras.losses.mean_squared_error(y_true, prediction)

        # Remove all NaN
        mse = train_util.meanWithoutNanOrInf(mse)

        mse = K.mean(mse)
        mse = mse * 100

        return mse
    return loss

def kpcnSSIM(noisy_img, kpcn_size):
    """Apply the kernel and use 1-ssim as loss function."""
    def loss(y_true, y_pred):
        prediction = train_util.applyKernel(noisy_img, y_pred, kpcn_size)
        # Normalise the images to ensure we don't get NaN
        y_true = y_true / K.max(y_true)
        prediction = prediction / K.max(prediction)
        ssim = tf.image.ssim(prediction, y_true, max_val=1.0)
        ssim = train_util.meanWithoutNanOrInf(1-ssim)
        return ssim
    return loss

def kpcnVGG(noisy_img, kpcn_size, vgg):
    """Apply the kernel and calulate VGG loss at block2_conv2."""
    def loss(y_true, y_pred):
        prediction = train_util.applyKernel(noisy_img, y_pred, kpcn_size)

        # Extract the features from prediction and ground truth
        features_pred = vgg(prediction)
        features_true = vgg(y_true)
        
        feature_loss = keras.losses.mean_squared_error(features_pred, features_true)
        feature_loss = K.mean(feature_loss)

        return feature_loss
    return loss

def noKernelVGG(vgg):
    def loss(y_pred, y_true):
        features_true = vgg(y_true)
        features_pred = vgg(y_pred)
        feature_loss = keras.losses.mean_squared_error(features_pred, features_true)
        return K.mean(feature_loss)
    return loss


def wassersteinLoss(y_true, y_pred):
    return K.mean(y_true * y_pred)
