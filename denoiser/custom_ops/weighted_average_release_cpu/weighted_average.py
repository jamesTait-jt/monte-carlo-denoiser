import tensorflow as tf
from math import sqrt
import numpy as np
from os import path

_module = tf.load_op_library(path.join(path.dirname(__file__), 'weighted_average_lib.so'))

@tf.RegisterShape("WeightedAverage")
def _weighted_average_shape(op):
    images  = op.inputs[0].get_shape()
    weights = op.inputs[1].get_shape()
    bs, w, h, c = images
    k = int(sqrt(int(weights[3]))) - 1
    return [(bs, w-k, h-k, c)]


@tf.RegisterGradient("WeightedAverage")
def _weighted_average_grad(op, grad):
    images = op.inputs[0]
    weights = op.inputs[1]
    grads = _module.weighted_average_gradients(weights, grad, images)
    grads = tf.clip_by_value(grads, -1000000, .1000000)
    return [None, grads]

weighted_average = _module.weighted_average

