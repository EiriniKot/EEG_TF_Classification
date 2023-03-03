import json
import numpy as np
from src.preprocessing import butter_bandpass_filter

from scipy.fft import fft, ifft
from sklearn import metrics
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Dropout, Flatten, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from src.utils import get_full_sample_paths, load_numpy, get_stratified_split

f = open('onehot_info.json')
label_index_info = json.load(f)
data_directory = 'data'

lowcut = 0.2
highcut = 60.0

if __name__ == '__main__':
    # Tuples with path and label index
    full_paths = get_full_sample_paths(data_directory)
    X, y = load_numpy(full_paths, label_index_info)

    for idx, i in enumerate(X):
        X[idx] = np.apply_along_axis(fft, 0, i)

    X = np.swapaxes(X, 1, 2)
    for i in range(0, len(X), 9):
        fig, axs = plt.subplots(3, 3)
        fig.tight_layout(pad=2.0)

        axs[0, 0].plot(X[i, :, :])
        axs[0, 0].set_title(y[i])

        axs[0, 1].plot(X[i + 1, :, :])
        axs[0, 1].set_title(y[i + 1])

        axs[0, 2].plot(X[i + 2, :, :])
        axs[0, 2].set_title(y[i + 2])

        axs[1, 0].plot(X[i + 3, :, :])
        axs[1, 0].set_title(y[i + 3])

        axs[1, 1].plot(X[i + 4, :, :])
        axs[1, 1].set_title(y[i + 4])

        axs[1, 2].plot(X[i + 5, :, :])
        axs[1, 2].set_title(y[i + 5])

        axs[2, 0].plot(X[i + 6, :, :])
        axs[2, 0].set_title(y[i + 6])

        axs[2, 1].plot(X[i + 7, :, :])
        axs[2, 1].set_title(y[i + 7])

        axs[2, 2].plot(X[i + 8, :, :])
        axs[2, 2].set_title(y[i + 8])

    X = butter_bandpass_filter(X, lowcut, highcut, fs=359./0.7, order=2)

    for idx, i in enumerate(X):
        X[idx] = np.apply_along_axis(ifft, 0, i)

    dict_sets = get_stratified_split(X, y, 4,
                                     train_size=0.8,
                                     val_size=0.1,
                                     test_size=0.1,
                                     random_state=0,
                                     one_hot=True)

    dict_sets['train'][0] = np.swapaxes(dict_sets['train'][0], 1, 2).astype(np.float64)
    dict_sets['val'][0] = np.swapaxes(dict_sets['val'][0], 1, 2).astype(np.float64)
    dict_sets['test'][0] = np.swapaxes(dict_sets['test'][0], 1, 2).astype(np.float64)

    batch_size = 28
    
    # Prepare the train dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((dict_sets['train'][0], 
                                                        dict_sets['train'][1]))
    train_dataset = train_dataset.shuffle(buffer_size=128).batch(batch_size)
    
    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((dict_sets['val'][0], 
                                                      dict_sets['val'][1]))
    val_dataset = val_dataset.batch(batch_size)
    
    # Prepare the test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((dict_sets['test'][0], 
                                                      dict_sets['test'][1]))
    test_dataset = test_dataset.batch(batch_size)
    
    model = tf.keras.Sequential(layers=[Conv1D(filters=128,
                                                kernel_size=3,
                                                strides=1,
                                                padding="same"),
                                        Normalization(),
                                        Dropout(0.1),
                                        MaxPool1D(pool_size=2,
                                                  padding="valid"),
                                        Conv1D(filters=32,
                                                kernel_size=3,
                                                strides=1,
                                                padding="valid"),
                                        Dropout(0.1),
                                        Flatten(),
                                        Dense(128, activation='relu'),
                                        Dense(4, activation='softmax')],
                                name='model_conv1d')
    
    # # Instantiate an optimizer.
    optimizer = Adam(learning_rate=1e-3)
    
    # Instantiate a loss function.
    loss_fn = CategoricalCrossentropy(label_smoothing=0.01)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['categorical_accuracy'])
    
    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    outs = model.evaluate(test_dataset)
    print('Loss, Accuracy', outs)

    print('Classification report')
    print(metrics.classification_report(dict_sets['test'][1],
                                        model.predict(dict_sets['test'][0])
                                        ))
