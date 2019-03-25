TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared weighted_average.cc -o weighted_average_lib.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

