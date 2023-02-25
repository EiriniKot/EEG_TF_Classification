import os
import numpy as np

from scipy import signal, fft


def apply_transform(arr, method='cwt', kwargs={}):
    if method == 'cwt':
        # Perform a continuous wavelet transform on data, using the wavelet function
        arr = np.apply_along_axis(lambda i: signal.cwt(i, **kwargs),
                                  axis=1,
                                  arr=arr)
    elif method == 'fft':
        # Perform a fft on the data
        arr = np.apply_along_axis(lambda i: fft.fft(i, **kwargs),
                                  axis=1,
                                  arr=arr)

    # Bring number of samples in the first dimension
    arr = np.swapaxes(arr, 0, 1)
    print(f'Input x shape after {method} ', arr.shape)
    return arr

