import tensorflow as tf
from tensorflow.keras.layers import Conv2D, InputLayer, MaxPool2D

model = tf.keras.Sequential(layers=[Conv2D(filters=32,
                                           kernel_size=3,
                                           strides=2,
                                           padding="same"),
                                    MaxPool2D(pool_size=2,
                                              padding="valid"),
                                    Conv2D(filters=32,
                                           kernel_size=3,
                                           strides=2,
                                           padding="same"),
                                    tf.keras.layers.Dense(4)],
                            name='model_conv')

