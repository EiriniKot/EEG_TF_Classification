import json
import numpy as np

from src.utils import get_full_sample_paths, load_numpy, get_stratified_split
from src.preprocessing import apply_scaling_per_channel

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # !IMPORTANT

f = open('onehot_info.json')
label_index_info = json.load(f)
data_directory = 'data'


channels = ['Iz','Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5',
            'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1',
            'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
            'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7',
            'PO3', 'O1', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz',
            'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6',
            'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
            'C2', 'C4', 'C6','T8', 'TP8', 'CP6', 'CP4', 'CP2',
            'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4']

indexes = []
keep = ['C3', 'Cz', 'C2', 'FC2']
for i in keep:
    indexes.append(channels.index(i))

if __name__ == '__main__':
    # Tuples with path and label index
    full_paths = get_full_sample_paths(data_directory)
    X, y = load_numpy(full_paths, label_index_info)
    X = X[:, indexes, :]

    dict_sets = get_stratified_split(X, y, 4,
                                     train_size=0.8,
                                     val_size=0.1,
                                     test_size=0.1,
                                     random_state=0,
                                     one_hot=True)

    out = list(map(lambda d: apply_scaling_per_channel(d[0], d[1], d[2]),
                   zip(np.swapaxes(dict_sets['train'][0], 0, 1),
                       np.swapaxes(dict_sets['val'][0], 0, 1),
                       np.swapaxes(dict_sets['test'][0], 0, 1))))

    out = list(zip(*out))
    dict_sets['train'][0] = np.swapaxes(np.array(out[0]), 0, 1).astype(np.float64)
    dict_sets['val'][0] = np.swapaxes(np.array(out[1]), 0, 1).astype(np.float64)
    dict_sets['test'][0] = np.swapaxes(np.array(out[2]), 0, 1).astype(np.float64)

    dddddd
    dict_sets['train'][0] = np.mean(dict_sets['train'][0], 1)
    dict_sets['val'][0] = np.mean(dict_sets['val'][0], 1)
    dict_sets['test'][0] = np.mean(dict_sets['test'][0], 1)
    print(dict_sets['train'][0].shape)
    gg
    batch_size = 48

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

    print('Final input shape ', dict_sets['train'][0].shape)

    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
          keras.layers.LSTM(
              units=128,
              input_shape=[dict_sets['train'][0].shape[1], dict_sets['train'][0].shape[2]]
          )
        )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))

    # # Instantiate an optimizer.
    optimizer = Adam(learning_rate=1e-3)

    # Instantiate a loss function.
    loss_fn = CategoricalCrossentropy(label_smoothing=0.001)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['categorical_accuracy'])

    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=100)

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
