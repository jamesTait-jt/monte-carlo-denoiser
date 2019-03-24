import tensorflow as tf

weighted_average_module = tf.load_op_library('./weighted_average.so')


with tf.Session(''):
    weights = tf.ones((7, 7, 5, 5))
    img = tf.ones((7, 7, 3))
    print(weighted_average_module.weighted_average(img, weights).eval())
