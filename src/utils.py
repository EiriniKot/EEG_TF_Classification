import os
import numpy as np


def get_full_sample_paths(data_directory):
    full_categ = [(os.path.join(data_directory, categ), categ) for categ in os.listdir(data_directory)]
    all_paths_labels = []
    for cat, label in full_categ:
        samples_path = os.listdir(cat)
        for sample in samples_path:
            all_paths_labels.append((os.path.join(cat, sample), label))
    return all_paths_labels


def load_numpy(full_paths, label_index_info):
    x = []
    y = []
    for sample_path, label in full_paths:
        x_sample = np.load(sample_path)
        y_sample = np.array(label_index_info[label])
        x.append(x_sample)
        y.append(y_sample)

    x = np.stack(x)
    y = np.stack(y)
    print('Input x shape ', x.shape)
    print('Target y shape ', y.shape)
    return x, y