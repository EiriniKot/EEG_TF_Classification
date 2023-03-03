import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


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

def get_stratified_split(X, y, N,
                         train_size=0.8, 
                         val_size=0.1,
                         test_size=0.1,
                         random_state=0,
                         one_hot=True):
    
    assert train_size+val_size+test_size==1
    
    sss = StratifiedShuffleSplit(n_splits=1, 
                                 train_size=train_size+val_size, 
                                 random_state=random_state)
    
    train_valid_index, test_index = list(sss.split(X, y))[0]
    X_t_v = X[train_valid_index]
    X_test = X[test_index]
    
    y_t_v = y[train_valid_index]
    y_test = y[test_index]
    
    
    sss = StratifiedShuffleSplit(n_splits=1, 
                                 train_size=train_size/(train_size+val_size), 
                                 random_state=1)
    train_index, valid_index = list(sss.split(X_t_v, y_t_v))[0]
    
    X_train = X[train_index]
    X_val = X[valid_index]
    
    y_train = y[train_index]
    y_val = y[valid_index]
    
    if one_hot:
        y_train = np.eye(N)[y_train]
        y_val = np.eye(N)[y_val]
        y_test = np.eye(N)[y_test]
        
    return {'train':[X_train, y_train],
            'val':[X_val, y_val],
            'test':[X_test, y_test]}
