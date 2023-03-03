import json
import numpy as np
from scipy.fft import fft, ifft

from src.utils import get_full_sample_paths, load_numpy, get_stratified_split
from src.preprocessing import apply_scaling_per_channel, butter_bandpass_filter

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

f = open('onehot_info.json')
label_index_info = json.load(f)
data_directory = 'data'

lowcut = 0.5
highcut = 50.0

if __name__ == '__main__':
    # Tuples with path and label index
    full_paths = get_full_sample_paths(data_directory)
    X, y = load_numpy(full_paths, label_index_info)
    p
    for idx, i in enumerate(X):
        X[idx] = np.apply_along_axis(fft, 0, i)
    print(X.shape)
    X = butter_bandpass_filter(X, lowcut, highcut, fs=359. / 0.7, order=7)

    for idx, i in enumerate(X):
        X[idx] = np.apply_along_axis(ifft, 0, i)

    dict_sets = get_stratified_split(X, y, 4,
                                     train_size=0.8,
                                     val_size=0.1,
                                     test_size=0.1,
                                     random_state=0,
                                     one_hot=False)

    out = list(map(lambda d: apply_scaling_per_channel(d[0], d[1], d[2]),
                   zip(np.swapaxes(dict_sets['train'][0], 0, 1),
                       np.swapaxes(dict_sets['val'][0], 0, 1),
                       np.swapaxes(dict_sets['test'][0], 0, 1))))

    out = list(zip(*out))
    dict_sets['train'][0] = np.swapaxes(np.array(out[0]), 0, 1).astype(np.float64)
    dict_sets['val'][0] = np.swapaxes(np.array(out[1]), 0, 1).astype(np.float64)
    dict_sets['test'][0] = np.swapaxes(np.array(out[2]), 0, 1).astype(np.float64)

    dict_sets['train'][0] = np.mean(dict_sets['train'][0], 1)
    dict_sets['val'][0] = np.mean(dict_sets['val'][0], 1)
    dict_sets['test'][0] = np.mean(dict_sets['test'][0], 1)

    rf_clf = RandomForestClassifier(max_depth=8, random_state=0)
    rf_clf.fit(dict_sets['train'][0], dict_sets['train'][1])
    out = rf_clf.score(dict_sets['test'][0], dict_sets['test'][1].astype(np.int32))

    print("Classification report")
    print(metrics.classification_report(dict_sets['test'][1].astype(np.int32),
                                  rf_clf.predict(dict_sets['test'][0])
                         ))

    knn_clf = KNeighborsClassifier(n_neighbors=12)
    knn_clf.fit(dict_sets['train'][0], dict_sets['train'][1].astype(np.int32))
    out = knn_clf.score(dict_sets['test'][0], dict_sets['test'][1].astype(np.int32))

    print('Classification report')
    print(metrics.classification_report(dict_sets['test'][1].astype(np.int32),
                                  knn_clf.predict(dict_sets['test'][0])
                         ))
    print('Accuracy KNeighborsClassifier test : ', out)


