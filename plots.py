import json
import numpy as np


from src.utils import get_full_sample_paths, load_numpy
from src.preprocessing import butter_bandpass_filter
from scipy.fft import fft, ifft

import matplotlib.pyplot as plt

f = open('onehot_info.json')
label_index_info = json.load(f)
data_directory = 'data'

lowcut = 5.
highcut = 60.0


if __name__ == '__main__':
    # Tuples with path and label index
    channels = ['Iz', 'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5',
                'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3',
                'O1', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4',
                'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
                'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6',
                'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4',
                'O2']
    idx_2_keep = channels.index('Cz')
    idx_2_keep = [13, 14, 47, 48, 49,50,51]
    
    full_paths = get_full_sample_paths(data_directory)
    X, y = load_numpy(full_paths, label_index_info)
    label = 0
    X = X[0:-1:15]
    X = np.take(X, idx_2_keep, axis=1)
    y = y[0:-1:15]
    
    for idx, i in enumerate(X):
        X[idx] = np.apply_along_axis(fft, 0, i)
    
    X = butter_bandpass_filter(X, lowcut, highcut, fs=359./0.7, order=7)  
    
    for idx, i in enumerate(X):
        X[idx] = np.apply_along_axis(ifft, 0, i)

    
    X = np.swapaxes(X, 1, 2)
    for i in range(0, len(X), 9):
        
        fig, axs = plt.subplots(3, 3)
        fig.tight_layout(pad=2.0)

        axs[0, 0].plot(X[i,:,:])
        axs[0, 0].set_title(y[i])
        
        axs[0, 1].plot(X[i+1,:,:])
        axs[0, 1].set_title(y[i+1])
        
        axs[0, 2].plot(X[i+2,:,:])
        axs[0, 2].set_title(y[i+2])
        
        axs[1, 0].plot(X[i+3,:,:])
        axs[1, 0].set_title(y[i+3])
        
        axs[1, 1].plot(X[i+4,:,:])
        axs[1, 1].set_title(y[i+4])
        
        axs[1, 2].plot(X[i+5,:,:])
        axs[1, 2].set_title(y[i+5])
        
        axs[2, 0].plot(X[i+6,:,:])
        axs[2, 0].set_title(y[i+6])
        
        axs[2, 1].plot(X[i+7,:,:])
        axs[2, 1].set_title(y[i+7])
        
        axs[2, 2].plot(X[i+8,:,:])
        axs[2, 2].set_title(y[i+8])
