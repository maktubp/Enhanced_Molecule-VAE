import tensorflow as tf

print("TF version:", tf.__version__)
print(tf.__file__)
tf.test.is_built_with_cuda()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
