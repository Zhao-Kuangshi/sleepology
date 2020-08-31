# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:59:08 2020

@author: chizh
"""
import math
import mne
import numpy
from . import dataset
from fractions import Fraction

def resample(data, to_sfreq, from_sfreq = None, axis = -1):
    if from_sfreq is None:
        from_sfreq = data.shape[axis]
    frac = Fraction(to_sfreq, from_sfreq)
    ret = mne.filter.resample(data, up = float(frac.numerator),
                               down = float(frac.denominator), axis = axis)
    return ret

def eeg_power_band(epochs, picks = 'eeg'):
    """脑电相对功率带特征提取
    该函数接受一个""mne.Epochs"对象，
    并基于与scikit-learn兼容的特定频带中的相对功率创建EEG特征。
    Parameters
    ----------
    epochs : Epochs
        The data.
    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # 特定频带
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    psds, freqs = mne.time_frequency.psd_welch(epochs, picks=picks)
    # 归一化 PSDs
    psds = normalize(psds, axis = -1)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return numpy.concatenate(X, axis=1)

def normalize(data, axis=None, log=False, atan=False, mean=0, sd=None):
    """
    Normalize an array to [0, 1] along an axis.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.
        
    axis : None or int, optional
        Axis or axes along which to operate. Normalization will  By default, flattened input is
        used. The default is None.
        
    atan : bool, optional
        

    Returns
    -------
    numpy.ndarray
        Normalized array to [0, 1] along an axis, whose maximum along the axis 
        should be 1 and minimum should be 0.
        
    Examples
    --------
    >>> a = numpy.arange(54).reshape((2,3,3,3))
    >>> a
    array([[[[ 0,  1,  2],
             [ 3,  4,  5],
             [ 6,  7,  8]],
    
            [[ 9, 10, 11],
             [12, 13, 14],
             [15, 16, 17]],
    
            [[18, 19, 20],
             [21, 22, 23],
             [24, 25, 26]]],
    
    
           [[[27, 28, 29],
             [30, 31, 32],
             [33, 34, 35]],
    
            [[36, 37, 38],
             [39, 40, 41],
             [42, 43, 44]],
    
            [[45, 46, 47],
             [48, 49, 50],
             [51, 52, 53]]]])
    
    >>> n = normalize(a, axis = -2)
    array([[[[0. , 0. , 0. ],
             [0.5, 0.5, 0.5],
             [1. , 1. , 1. ]],
    
            [[0. , 0. , 0. ],
             [0.5, 0.5, 0.5],
             [1. , 1. , 1. ]],
    
            [[0. , 0. , 0. ],
             [0.5, 0.5, 0.5],
             [1. , 1. , 1. ]]],
    
    
           [[[0. , 0. , 0. ],
             [0.5, 0.5, 0.5],
             [1. , 1. , 1. ]],
    
            [[0. , 0. , 0. ],
             [0.5, 0.5, 0.5],
             [1. , 1. , 1. ]],
    
            [[0. , 0. , 0. ],
             [0.5, 0.5, 0.5],
             [1. , 1. , 1. ]]]])
    
    >>> n.max(-2)
    array([[[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]])
    
    >>> n.min(-2)
    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]])
    
    >>> n = normalize(a, axis = -1)
    
    >>> n
    array([[[[0. , 0.5, 1. ],
             [0. , 0.5, 1. ],
             [0. , 0.5, 1. ]],
    
            [[0. , 0.5, 1. ],
             [0. , 0.5, 1. ],
             [0. , 0.5, 1. ]],
    
            [[0. , 0.5, 1. ],
             [0. , 0.5, 1. ],
             [0. , 0.5, 1. ]]],
    
    
           [[[0. , 0.5, 1. ],
             [0. , 0.5, 1. ],
             [0. , 0.5, 1. ]],
    
            [[0. , 0.5, 1. ],
             [0. , 0.5, 1. ],
             [0. , 0.5, 1. ]],
    
            [[0. , 0.5, 1. ],
             [0. , 0.5, 1. ],
             [0. , 0.5, 1. ]]]])
    
    >>> n.max(-1)
    array([[[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]])
    
    >>> n.min(-1)
    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]])
    
    >>> a[1][2][2][2] = 1024000

    >>> a
    array([[[[      0,       1,       2],
             [      3,       4,       5],
             [      6,       7,       8]],
    
            [[      9,      10,      11],
             [     12,      13,      14],
             [     15,      16,      17]],
    
            [[     18,      19,      20],
             [     21,      22,      23],
             [     24,      25,      26]]],
    
    
           [[[     27,      28,      29],
             [     30,      31,      32],
             [     33,      34,      35]],
    
            [[     36,      37,      38],
             [     39,      40,      41],
             [     42,      43,      44]],
    
            [[     45,      46,      47],
             [     48,      49,      50],
             [     51,      52, 1024000]]]])

    >>> normalize(a)
    array([[[[0.00000000e+00, 9.76562500e-07, 1.95312500e-06],
             [2.92968750e-06, 3.90625000e-06, 4.88281250e-06],
             [5.85937500e-06, 6.83593750e-06, 7.81250000e-06]],
    
            [[8.78906250e-06, 9.76562500e-06, 1.07421875e-05],
             [1.17187500e-05, 1.26953125e-05, 1.36718750e-05],
             [1.46484375e-05, 1.56250000e-05, 1.66015625e-05]],
    
            [[1.75781250e-05, 1.85546875e-05, 1.95312500e-05],
             [2.05078125e-05, 2.14843750e-05, 2.24609375e-05],
             [2.34375000e-05, 2.44140625e-05, 2.53906250e-05]]],
    
    
           [[[2.63671875e-05, 2.73437500e-05, 2.83203125e-05],
             [2.92968750e-05, 3.02734375e-05, 3.12500000e-05],
             [3.22265625e-05, 3.32031250e-05, 3.41796875e-05]],
    
            [[3.51562500e-05, 3.61328125e-05, 3.71093750e-05],
             [3.80859375e-05, 3.90625000e-05, 4.00390625e-05],
             [4.10156250e-05, 4.19921875e-05, 4.29687500e-05]],
    
            [[4.39453125e-05, 4.49218750e-05, 4.58984375e-05],
             [4.68750000e-05, 4.78515625e-05, 4.88281250e-05],
             [4.98046875e-05, 5.07812500e-05, 1.00000000e+00]]]])

    >>> normalize(a, log = True)
    array([[[[0.        , 0.05008568, 0.07938393],
             [0.10017136, 0.11629535, 0.12946961],
             [0.14060829, 0.15025705, 0.15876786]],
    
            [[0.16638104, 0.17326799, 0.17955529],
             [0.18533905, 0.19069397, 0.19567928],
             [0.20034273, 0.20472337, 0.20885354]],
    
            [[0.21276035, 0.21646672, 0.21999222],
             [0.22335368, 0.22656569, 0.22964098],
             [0.23259071, 0.23542473, 0.23815179]]],
    
    
           [[[0.24077965, 0.24331529, 0.24576496],
             [0.2481343 , 0.25042841, 0.25265192],
             [0.25480905, 0.25690364, 0.25893922]],
    
            [[0.26091903, 0.26284603, 0.26472298],
             [0.2665524 , 0.26833665, 0.2700779 ],
             [0.27177817, 0.27343936, 0.27506321]],
    
            [[0.27665137, 0.27820537, 0.27972666],
             [0.28121657, 0.28267639, 0.28410729],
             [0.28551041, 0.2868868 , 1.        ]]]])
    
    """
    if log and atan:
        raise TypeError('Parameter `log` and `atan` cannot be both true.')
    elif log:
        data_max = data.max(axis = axis, keepdims = True)
        data_min = data.min(axis = axis, keepdims = True)
        data = data - data_min + 1
        data = numpy.log10(data) / numpy.log10(data_max)
    elif atan:
        data_max = data.max(axis = axis, keepdims = True)
        data_min = data.min(axis = axis, keepdims = True)
        data -= mean
        if sd:
            # Scaling data to [-2, 2], where arctangent function is sensitive.
            data = data / abs(sd) * 2
        data = numpy.arctan(data)
    else:
        pass

    data_max = data.max(axis = axis, keepdims = True)
    data_min = data.min(axis = axis, keepdims = True)

    ret = (data - data_min) / (data_max - data_min)
    return ret
        
if __name__ == '__main__':
    
    ds = dataset.Dataset('tfr', '/userhome/result')
    ds.add_data('/userhome/data/', 'edf', 'edf', 
               '/userhome/label/', 'nihonkohden_txt', 'txt')
    ds.load_data()
    procedure = dataset.Procedure(preprocess_0308)
    procedure.add_channels(['C3-A1', 'C4-A2', 'O1-A2', 'O2-A1', 'EOG-L', 'EOG-R', 'Chin'])
    ds.preprocess(procedure)
    ds.exclude_data(['L', '0', 'o', 'O', 'S', 's'])
    ds.save()
    