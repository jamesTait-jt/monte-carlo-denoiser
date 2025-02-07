TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o weighted_average_gpu.cu.o weighted_average_gpu.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o weighted_average_gpu.so weighted_average_gpu.cc \
  weighted_average_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64
