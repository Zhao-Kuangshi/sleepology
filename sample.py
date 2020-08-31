# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:07:43 2020

@author: 赵匡是
"""

import six
import json
import numpy as np
import tableprint as tp
from sklearn.model_selection import ShuffleSplit

def pad_sequences(sequences, maxlen=None, dtype='float32',
                  padding='pre', truncating='pre', value=-1.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


class Sample(object):
    def __init__(self, unit='epoch', tmin=0, tmax=0, n_splits=10,
                 test_size=0.1, epoch_padding=False, data_padding=None):
        self.set_unit(unit)
        self.set_timespan(tmin, tmax)
        self.set_n_splits(n_splits)
        self.set_test_size(test_size)
        self.set_epoch_padding(epoch_padding)
        if data_padding is None:
            if self.get_unit() == 'data':
                self.set_data_padding(True)
            else:
                self.set_data_padding(False)
        else:
            self.set_data_padding(data_padding)

    def set_unit(self, unit):
        '''
        Set the primary unit of sampling process. 
        
        The `Sample` object is to sample \'data\' and \'labels\' from given
        `Dataset` object,i.e. the `x`s and `y`s of a model. There are a lot
        of ways. 
        
        For example, using data of one epoch as `x` and the state of that
        epoch as `y`. Relatively, we don't care about which data that epoch
        come from. So the \'primary unit\' in this situation is \'epoch\'.
        
        In another example, we use a series of epochs as `x`, and just one
        label for the series. Epochs are not divisible in this situation. So
        the \'primary unit\' is \'data\'.
        
        If `unit == \'epoch\'` , the first dimension of sampled `x` will be
        \'epoch\'. Then if you use `condition` as sampled `y`, 
        '''
        self.unit = unit

    def get_unit(self):
        return self.unit

    def set_timespan(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax

    def get_timespan(self):
        return (self.get_tmin(), self.get_tmax())

    def get_tmax(self):
        return self.tmax

    def get_tmin(self):
        return self.tmin

    def set_n_splits(self, n_splits):
        '''
        Set the iterations of dataset.

        Parameters
        ----------
        n_splits : int
            Number of re-shuffling & splitting iterations.
            If you don\'t want to cross validation, you can set `n_splits=1`.

        '''
        self.n_splits = n_splits

    def get_n_splits(self):
        return self.n_splits

    def set_test_size(self, test_size):
        '''
        Set size(proportion) of the test set.

        Parameters
        ----------
        test_size : float
            Should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split.

        '''
        self.test_size = test_size

    def get_test_size(self):
        return self.test_size

    def set_epoch_padding(self, epoch_padding):
        self.epoch_padding = epoch_padding

    def set_data_padding(self, data_padding, max_len = None):
        if self.unit == 'epoch':
            self.data_padding = False
        elif self.unit == 'data':
            self.data_padding = True
        self.max_len = max_len
        
    def set_x(self, x):
        # 不强求类型。可以是dataset的label甚至是condition(比如用很多condition来判断疾病(age, sex)→diagnose
        if isinstance(x, str):
            self.x = [x]
        elif isinstance(feature, list):
            self.x = x

    def get_x(self):
        try:
            return self.x
        except:
            return None
            
    def set_y(self, y):
        # unit是data就不是epoched，unit是epoch就是epoched。或者说，如果unit是data，就从condition里找，如果unit是epoch就从label里找。
        if isinstance(y, str):
            self.y = [y]
        elif isinstance(y, list):
            self.y = y

    def get_y(self):
        try:
            return self.y
        except:
            return None

    def summary(self):
        # 打印整个sample的规则
        tp.banner('Your Sampling Settings')
        headers = ['Parameter', 'Value']
        content = [['unit', self.get_unit()],
                   ['tmin', self.get_tmin],
                   ['tmax', self.get_tmax()],
                   ['n_splits', self.get_n_splits],
                   ['test_size', self.get_test_size()],
                   ['epoch_padding', 'True' if self.epoch_padding else 'False'],
                   ['data_padding', 'True' if self.data_padding else 'False'],
                   ['max_len', self.max_len],
                   ['x', self.get_x()],
                   ['y', self.get_y()]]
        tp.table(content, headers)

    @staticmethod
    def load(fpath):
        content = None
        with open(fpath, 'r') as f:
            content = json.load(f)
        sample = Sample(unit=content['unit'], tmin=content['tmin'],
                        tmax=content['tmax'], n_splits=content['n_splits'],
                        test_size=content['test_size'], 
                        epoch_padding=content['epoch_padding'],
                        data_padding=content['data_padding'])
        sample.set_data_padding(content['data_padding'],
                                max_len= content['max_len'])
        sample.set_x(content['x'])
        sample.set_y(content['y'])
        return sample

    def save(fpath):
        content = {'unit' : self.get_unit(),
                   'tmin' : self.get_tmin,
                   'tmax' : self.get_tmax(),
                   'n_splits' : self.get_n_splits,
                   'test_size' : self.get_test_size(),
                   'epoch_padding' : 'True' if self.epoch_padding else 'False',
                   'data_padding' : 'True' if self.data_padding else 'False',
                   'max_len' : self.max_len,
                   'x' : self.get_x(),
                   'y' : self.get_y()}
        with open(fpath, 'w') as f:
            json.dump(content, f)

    def check_dataset(self):
        # check `x`
        if self.get_x() is None:
            raise AssertionError('`x` hasn\'t set. Please use `set_x` first.')
        else:
            for item in self.get_x():
                # check if `x` in dataset.elements
                if item not in self.dataset.elements.keys():
                    raise AssertionError('The target dataset has no element `'+
                                         item + '`.')
                # while unit == `epoch`, `x` cannot be `condition` (because
                # `condition` is not epoched feature, cannot represent an epoch)
                elif self.get_unit() == 'epoch' and \
                    self.dataset.elements[item] == 'condition':
                    raise AssertionError('while unit == `epoch`, `x` cannot be'
                                         ' `condition` (because `condition` is'
                                         ' not epoched feature, cannot'
                                         ' represent an epoch)')
        # check `y`
        if self.mode == 'train':  # in training mode, `y` must provided
            if self.get_y() is None:
                raise AssertionError('`y` hasn\'t set.'
                                     'Please use `set_y` first.')
            else:
                for item in self.get_y():
                    # check if `y` in dataset.elements
                    if item not in self.dataset.elements.keys():
                        raise AssertionError('The target dataset has no'
                                             ' element `' + item + '`.')
                    # check if `y` is `label` or `condition`
                    elif self.dataset.elements[item] != 'label' or \
                        self.dataset.element[item] != 'condition':
                        raise AssertionError('The `y` must be `label` or'
                                             ' `condition`. But ' + item +
                                             'was a `' +
                                             self.dataset.elements[item] +'`.')
                    # while unit == `data`, `y` must be `condition` (because
                    # `label` is epoched label, cannot represent a data)
                    elif self.get_unit() == 'data' and \
                        self.dataset.elements[item] == 'label':
                            raise AssertionError('while unit == `data`, `y` '
                                                 'must be `condition` (because'
                                                 ' `label` is epoched label, '
                                                 'cannot represent a data)')
        # and in prediction mode, `y` will not be used

    def from_dataset(self, dataset, data_selection = None, mode = 'train'):
        # print the summary
        self.summary()

        # set mode
        if mode.lower() == 'train':
            self.mode = 'train'
        elif mode.lower() == 'predict':
            self.mode = 'predict'
            if self.data_padding and self.max_len is None:
                raise ValueError('In mode `predict`, max_len of data_padding'+
                                 ' must be set.')
        else:
            raise ValueError('Invalid mode. Must be \'train\' or \'predict\'.')

        # check dataset
        self.check_dataset()

        # set dataset
        self.dataset = dataset
        if data_selection is None and self.unit == 'epoch':
            self.data_selection = self.dataset.select_epochs()
        elif data_selection is None and self.unit == 'data':
            self.data_selection = self.dataset.select_data()
        else:
            self.data_selection = data_selection

        # set max_len
        if self.data_padding and self.max_len is None:
            epochs = self.dataset.epochs_per_data(self.data_list())
            self.max_len = max(epochs)

    def data_list(self):
        # TODO : self.data_selection不再是dict了！变成二元组(data_name, epoch)
        rst = set()
        for item in self.data_selection:
            if isinstance(item, str):
                rst.add(item)
            else:
                rst.add(item[0])
        return list(rst)

    def next_subgroup():
        # 
        pass

然后就是各种generator、padding