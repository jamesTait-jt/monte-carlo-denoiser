import tensorflow as tf
from math import sqrt
from os import path
from tensorflow.python.framework import ops

# weighted_average.py: Calls library that efficiently computes the weighted average

_module = tf.load_op_library(path.join(path.dirname(__file__), "./custom_ops/weighted_average.so"))
#_module = tf.load_op_library(path.join(path.dirname(__file__), "./custom_ops/weighted_average_gpu.so"))

@tf.RegisterGradient("WeightedAverage")
def _weighted_average_grad(op, grad):
    images = op.inputs[0]
    weights = op.inputs[1]
    grads = _module.weighted_average_gradients(weights, grad, images)
    grads = tf.clip_by_value(grads, -0.05000000, 0.05000000)
    return [None, grads]


weighted_average = _module.weighted_average
