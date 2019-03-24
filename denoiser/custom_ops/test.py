import tensorflow as tf

weighted_average_module = tf.load_op_library('./weighted_average.so')


with tf.Session(''):
    weights = tf.ones((25, 64, 64, 21, 21))
    img = tf.ones((25, 64, 64, 3))
    print(weighted_average_module.weighted_average(img, weights).eval())
