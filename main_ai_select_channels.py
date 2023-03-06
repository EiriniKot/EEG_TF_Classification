import json
import numpy as np
from scipy.fft import fft, ifft

from src.utils import get_full_sample_paths, load_numpy, get_stratified_split
from src.preprocessing import apply_scaling_per_channel, butter_bandpass_filter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # !IMPORTANT

f = open('onehot_info.json')
label_index_info = json.load(f)
data_directory = 'data'

lowcut = 0.5
highcut = 50.0

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
    X = X[:,indexes,:]
    dict_sets = get_stratified_split(X, y, 4,
                                     train_size=0.89,
                                     val_size=0.01,
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

    feature = 'mean'

    if feature=='mean':
        dict_sets['train'][0] = np.mean(dict_sets['train'][0], 1)
        dict_sets['val'][0] = np.mean(dict_sets['val'][0], 1)
        dict_sets['test'][0] = np.mean(dict_sets['test'][0], 1)
    elif feature=='max':
        dict_sets['train'][0] = np.max(dict_sets['train'][0], 1)
        dict_sets['val'][0] = np.max(dict_sets['val'][0], 1)
        dict_sets['test'][0] = np.max(dict_sets['test'][0], 1)

    # print(dict_sets['train'][0].shape)
    # for i in range(0, len(dict_sets['train'][0]), 9):
    #     fig, axs = plt.subplots(3, 3)
    #     fig.tight_layout(pad=2.0)
    #     axs[0, 0].plot(dict_sets['train'][0][i, :])
    #     axs[0, 0].set_title(y[i])
    #
    #     axs[0, 1].plot(dict_sets['train'][0][i + 1, :])
    #     axs[0, 1].set_title(y[i + 1])
    #
    #     axs[0, 2].plot(dict_sets['train'][0][i + 2, :])
    #     axs[0, 2].set_title(y[i + 2])
    #
    #     axs[1, 0].plot(dict_sets['train'][0][i + 3, :])
    #     axs[1, 0].set_title(y[i + 3])
    #
    #     axs[1, 1].plot(dict_sets['train'][0][i + 4, :])
    #     axs[1, 1].set_title(y[i + 4])
    #
    #     axs[1, 2].plot(dict_sets['train'][0][i + 5, :])
    #     axs[1, 2].set_title(y[i + 5])
    #
    #     axs[2, 0].plot(dict_sets['train'][0][i + 6, :])
    #     axs[2, 0].set_title(y[i + 6])
    #
    #     axs[2, 1].plot(dict_sets['train'][0][i + 7, :])
    #     axs[2, 1].set_title(y[i + 7])
    #
    #     axs[2, 2].plot(dict_sets['train'][0][i + 8, :])
    #     axs[2, 2].set_title(y[i + 8])
    #     plt.show()
    #     fff
    rf_clf = RandomForestClassifier(max_depth=11, random_state=0)
    rf_clf.fit(dict_sets['train'][0], dict_sets['train'][1])
    out = rf_clf.score(dict_sets['test'][0], dict_sets['test'][1].astype(np.int32))

    print("Classification report")
    print(metrics.classification_report(dict_sets['test'][1].astype(np.int32),
                                  rf_clf.predict(dict_sets['test'][0])
                         ))
    print('Accuracy Random Forest Classifier test : ', out)
    cf = metrics.confusion_matrix(dict_sets['test'][1],
                                  rf_clf.predict(dict_sets['test'][0])
                                  )
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cf)

    cm_display.plot()
    plt.title('Confusion Matrix  Random Forests')
    plt.show()

    knn_clf = KNeighborsClassifier(n_neighbors=10)
    knn_clf.fit(dict_sets['train'][0], dict_sets['train'][1].astype(np.int32))
    out = knn_clf.score(dict_sets['test'][0], dict_sets['test'][1].astype(np.int32))

    print('Classification report')
    print(metrics.classification_report(dict_sets['test'][1].astype(np.int32),
                                  knn_clf.predict(dict_sets['test'][0])
                         ))
    print('Accuracy KNeighborsClassifier test : ', out)

    cf = metrics.confusion_matrix(dict_sets['test'][1],
                                  knn_clf.predict(dict_sets['test'][0])
                                  )
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cf)

    cm_display.plot()
    plt.title('Confusion Matrix  KNeighbors')
    plt.show()

    ada = AdaBoostClassifier(n_estimators=4000, random_state=0)
    ada.fit(dict_sets['train'][0], dict_sets['train'][1])
    out = ada.score(dict_sets['test'][0], dict_sets['test'][1].astype(np.int32))

    print("Classification report")
    print(metrics.classification_report(dict_sets['test'][1].astype(np.int32),
                                        ada.predict(dict_sets['test'][0])
                                        ))
    print('Accuracy AdaBoost Classifier test : ', out)
    cf = metrics.confusion_matrix(dict_sets['test'][1],
                                  ada.predict(dict_sets['test'][0])
                                  )
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cf)

    cm_display.plot()
    plt.title('Confusion Matrix AdaBoost')
    plt.show()

