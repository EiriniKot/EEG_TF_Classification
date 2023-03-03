import os
import numpy as np

from scipy import signal, fft
from scipy.signal import butter, lfilter

from sklearn.preprocessing import MinMaxScaler


def apply_transform(arr, method='cwt', kwargs={}):
    if method == 'cwt':
        # Perform a continuous wavelet transform on data, using the wavelet function
        arr = np.apply_along_axis(lambda i: signal.cwt(i, **kwargs),
                                  axis=0,
                                  arr=arr)
        
        # Bring number of samples in the first dimension
        arr = np.swapaxes(arr, 0, 1)
        
    elif method == 'fft':
        # Perform a fft on the data
        arr = np.apply_along_axis(lambda i: fft.fft(i, **kwargs),
                                  axis=0,
                                  arr=arr)
        
    elif method == 'stft':
        # Perform a fft on the data
        arr = np.apply_along_axis(lambda i: signal.stft(i, **kwargs),
                                  axis=0,
                                  arr=arr)

    
    print(f'Input x shape after {method} ', arr.shape)
    return arr

def apply_scaling_per_channel(train_data, valid_data, test_data):
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)
    return train_data, valid_data, test_data


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y