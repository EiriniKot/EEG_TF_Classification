import json
import numpy as np

from scipy.signal import cwt, ricker, morlet
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # !IMPORTANT

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Dropout, Flatten, Normalization, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from src.utils import get_full_sample_paths, load_numpy, get_stratified_split
from src.preprocessing import apply_scaling_per_channel


f = open('onehot_info.json')
label_index_info = json.load(f)
data_directory = 'data'

lowcut = 0.00002
highcut = 0.1

if __name__ == '__main__':
    # Tuples with path and label index
    full_paths = get_full_sample_paths(data_directory)
    X, y = load_numpy(full_paths, label_index_info)

    dict_sets = get_stratified_split(X, y, 4,
                                     train_size=0.8,
                                     val_size=0.1,
                                     test_size=0.1,
                                     random_state=0,
                                     one_hot=True)


    feature = 'mean'

    if feature == 'mean':
        dict_sets['train'][0] = np.mean(dict_sets['train'][0], 1)
        dict_sets['val'][0] = np.mean(dict_sets['val'][0], 1)
        dict_sets['test'][0] = np.mean(dict_sets['test'][0], 1)

    cwt_s = []
    widths = np.arange(1, 30, 6)

    for idx, i in enumerate(dict_sets['train'][0]):
        cwt_s.append(cwt(i, wavelet=morlet, widths=widths))
    dict_sets['train'][0] = np.stack(cwt_s, 0).astype(np.float64)

    cwt_s = []
    for idx, i in enumerate(dict_sets['val'][0]):
        cwt_s.append(np.apply_along_axis(lambda j: cwt(j, wavelet=ricker, widths=widths), 0, i))
    dict_sets['val'][0] = np.stack(cwt_s, 0).astype(np.float64)

    cwt_s = []
    for idx, i in enumerate(dict_sets['test'][0]):
        cwt_s.append(np.apply_along_axis(lambda j: cwt(j, wavelet=ricker, widths=widths), 0, i))
    dict_sets['test'][0] = np.stack(cwt_s, 0).astype(np.float64)

    dict_sets['train'][0] = np.swapaxes(dict_sets['train'][0], 1, 2)
    dict_sets['val'][0] = np.swapaxes(dict_sets['val'][0], 1, 2)
    dict_sets['test'][0] = np.swapaxes(dict_sets['test'][0], 1, 2)

    batch_size = 286

    # Prepare the train dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((dict_sets['train'][0],
                                                        dict_sets['train'][1]))
    train_dataset = train_dataset.shuffle(buffer_size=21).batch(batch_size)

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
                                               activation='relu',
                                               padding="same"),
                                        Normalization(),
                                        Dropout(0.15),
                                        AveragePooling1D(pool_size=3,
                                                         padding="valid"),
                                        Normalization(),
                                        Dropout(0.15),
                                        Flatten(),
                                        Dense(128, activation='relu'),
                                        Dense(4, activation='softmax')],
                                name='model_conv1d')

    # # Instantiate an optimizer.
    optimizer = Adam(learning_rate=1e-3)

    # Instantiate a loss function.
    loss_fn = CategoricalCrossentropy(label_smoothing=0.001)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['categorical_accuracy'])

    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=50)

    outs = model.evaluate(test_dataset)
    print('Loss, Accuracy', outs)

    print('Classification report')
    print(model.predict(dict_sets['test'][0]))
    print(metrics.classification_report(np.argmax(dict_sets['test'][1], 1),
                                        np.argmax(model.predict(dict_sets['test'][0]), 1)
                                        ))

    cf = metrics.confusion_matrix(np.argmax(dict_sets['test'][1], 1),
                                  np.argmax(model.predict(dict_sets['test'][0]), 1))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cf)

    cm_display.plot()
    plt.show()

