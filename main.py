import json
import numpy as np
from src.preprocessing import apply_transform

from scipy.signal import morlet2

# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, InputLayer, MaxPool2D
# from tensorflow.keras import optimizers

from src.utils import get_full_sample_paths, load_numpy

f = open('onehot_info.json')
label_index_info = json.load(f)
data_directory = 'data'

if __name__ == '__main__':
    full_paths = get_full_sample_paths(data_directory)
    x, y = load_numpy(full_paths, label_index_info)
    x = apply_transform(arr=x, method='cwt', kwargs={'widths': np.arange(1, 10),
                                                     'wavelet': morlet2})


    # batch_size = 10
    #
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    #
    # # Prepare the validation dataset.
    # val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    # val_dataset = val_dataset.batch(batch_size)
    #
    # # Prepare the test dataset.
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # test_dataset = test_dataset.batch(batch_size)
    #
    # x = np.apply_along_axis(fft.fft,
    #                         axis=1,
    #                         arr=x)
    #
    # print(x[0][0])
    #
    # model = tf.keras.Sequential(layers=[Conv2D(filters=32,
    #                                            kernel_size=3,
    #                                            strides=2,
    #                                            padding="same"),
    #                                     MaxPool2D(pool_size=2,
    #                                               padding="valid"),
    #                                     Conv2D(filters=32,
    #                                            kernel_size=3,
    #                                            strides=2,
    #                                            padding="same"),
    #                                     tf.keras.layers.Dense(4)],
    #                             name='model_conv')
    # # Instantiate an optimizer.
    # optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # # Instantiate a loss function.
    # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # model.compile()
    #