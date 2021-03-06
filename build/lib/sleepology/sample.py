# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:07:43 2020

@author: 赵匡是
"""
from .exceptions import ModeError, LackOfParameterError, BrokenTimestepError,\
    KFoldError, LackOfLabelDictError, TaskError
from .labeldict import BaseDict

import math
import json
import random
import logging
import numpy as np
import tableprint as tp
from sklearn.model_selection import ShuffleSplit
from typing import Dict, Union

def flatten(seq):
    return [i for a in seq for i in a]

class Sample(object):
    def __init__(self, unit='epoch', tmin=0, tmax=0, n_splits=10,
                 test_size=0.1, class_balance=True, data_balance=False,
                 epoch_padding=False, data_padding=None,
                 task='classification', concat=False):
        self.__editable = True
        self.__autoencoder = False
        self.set_unit(unit)
        self.set_timespan(tmin, tmax)
        self.set_n_splits(n_splits)
        self.set_test_size(test_size)
        self.set_class_balance(class_balance)
        self.set_data_balance(data_balance)
        self.set_epoch_padding(epoch_padding)
        self.set_data_padding(data_padding)
        self.set_task(task)
        self.array_type = int
        self.concat = concat

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
        self.__check_editable()
        self.unit = unit

    def get_unit(self):
        return self.unit

    def set_timespan(self, tmin, tmax):
        self.__check_editable()
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
        self.__check_editable()
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
        self.__check_editable()
        self.test_size = test_size

    def get_test_size(self):
        return self.test_size

    def set_class_balance(self, class_balance):
        self.__check_editable()
        self.class_balance = class_balance

    def get_class_balance(self):
        return self.class_balance

    def set_data_balance(self, data_balance):
        self.__check_editable()
        if data_balance and self.unit == 'data':
            self.data_balance = False
            raise ValueError('you cannot set `data_balance` as True while '
                             'primary unit is `data`. Please use '
                             '`Sample.set_unit(\'epoch\')` first, then use '
                             '`Sample.set_data_balance(True)` again')
        else:
            self.data_balance = data_balance

    def get_data_balance(self):
        return self.data_balance

    def set_epoch_padding(self, epoch_padding):
        self.__check_editable()
        self.epoch_padding = epoch_padding

    def set_data_padding(self, data_padding, max_len = None):
        self.__check_editable()
        if data_padding is None:
            if self.unit == 'epoch':
                self.data_padding = False
            elif self.unit == 'data':
                self.data_padding = True
        else:
            self.data_padding = data_padding
        self.max_len = max_len

    def set_task(self, task):
        '''
        Set task you want to train. `'classification'`, `regression` and
        `autoencoder` is valid. Different task types use different sample
        mechanisms. For example, in `regression` task, we have no idea to do
        `class_balance`.

        Parameters
        ----------
        task : str
            'classification', 'regression' or 'autoencoder'.

        Raises
        ------
        ValueError
            Invalid parameter.

        '''
        candidate = ['classification', 'regression', 'autoencoder']
        # check tsk
        if task.lower() not in candidate:
            raise ValueError('invalid task.')
        if task == 'autoencoder':
            self.__autoencoder = True
            self.task = 'regression'
        else:
            self.task = task.lower()

    def set_label_array_type(self, array_type:type):
        '''
        Set the type of one-hot or N-hot label array the sample will return.
        If `array_type` is `None`, the function will return a number directly,
        i.e., not in terms of one-hot or N-hot array.

        Parameters
        ----------
        array_type : type or None, optional
            The type of elements in return array. The default and most
            recommended is int. You should input directly a `type`, not a
            `str`.
            For example:
                array_type=bool          (Correct)
                array_type='bool'        (Wrong)

            If `array_type` is `None`, the function will return a number
            directly, i.e., not in terms of one-hot or N-hot array.

        Raises
        ------
        TypeError
            Raised when you input an empty `label` parameter.

        Returns
        -------
        int or np.ndarray
            return `int` if array_type is None.
            else return one-hot or N-hot `np.ndarray`. 

        '''
        self.array_type = array_type

    def set_concat(self, concat:bool=True):
        '''
        Decide the output shape when tmin!=0 or tmax!=0, i.e., multiple time
        steps are sampled.
        If you have not set the `tmin` or `tmax`. You do not need to care about
        this parameter.
        The default is False, means the output will not concatenate time steps
        (i.e. sampled epochs) but leave the output shape as
                   `(time_step, channel[, feature-related])`
        In this situation, every
        time step occupies a unique matrix. It is suitable for RNN-related
        models.
        If `concat==True`, all the sampled epoches will concatenate by the
        'time' axis. It results in continuous feature which is suitable for TCN
        or other CNN-related models.

        Parameters
        ----------
        concat : bool, optional
            The default is True.

        '''
        self.concat = concat

    def set_x(self, x, labeldict: Dict[str, BaseDict] = {}):
        # 不强求类型。可以是dataset的label甚至是condition(比如用很多condition来判断疾病(age, sex)→diagnose
        self.__check_editable()
        if isinstance(x, str):
            self.x = [x]
        elif isinstance(x, list):
            self.x = x
        if self.__autoencoder:
            self.set_y(self.get_x())
        self.x_dict = labeldict

    def get_x(self):
        try:
            return self.x
        except:
            return None
            
    def set_y(self, y, labeldict: Dict[str, BaseDict] = {}):
        # unit是data就不是epoched，unit是epoch就是epoched。或者说，如果unit是data，就从condition里找，如果unit是epoch就从label里找。
        self.__check_editable()
        if isinstance(y, str):
            y = [y]
        elif isinstance(y, list):
            pass
        if self.__autoencoder and self.get_x() is not None and y != \
            self.get_x(): # assert x == y when using autoencoder
            raise ValueError('When using autoencoder, `y` must be `x` itself.')
        self.y = y
        self.y_dict = labeldict

    def get_y(self):
        try:
            return self.y
        except:
            return None

    def summary(self):
        # 打印整个sample的规则
        headers = ['Parameter', 'Value']
        content = [['unit', self.get_unit()],
                   ['tmin', self.get_tmin()],
                   ['tmax', self.get_tmax()],
                   ['n_splits', self.get_n_splits()],
                   ['test_size', self.get_test_size()],
                   ['class_balance', 'True' if self.get_class_balance() else
                    'False'],
                   ['data_balance', 'True' if self.get_data_balance() else
                    'False'],
                   ['epoch_padding', 'True' if self.epoch_padding else
                    'False'],
                   ['data_padding', 'True' if self.data_padding else 'False'],
                   ['max_len', self.max_len if self.max_len is not None else
                    'Not Set'],
                   ['task', self.task],
                   ['x', str(self.get_x())],
                   ['y', str(self.get_y())],
                   ['Sample Mode', self.mode if hasattr(self, 'mode') else 
                    'Not Set'],
                   ['label_array_type', '{0}: {1}'.format(
                       'one-hot/N-hot' if self.array_type else 'numeric',
                       self.array_type if self.array_type else 'int'
                   )]]
        tp.banner('Your Sampling Settings')
        tp.table(content, headers)

    @staticmethod
    def load(fpath):
        content = None
        with open(fpath, 'r') as f:
            content = json.load(f)
        sample = Sample(unit=content['unit'], tmin=content['tmin'],
                        tmax=content['tmax'], n_splits=content['n_splits'],
                        test_size=content['test_size'],
                        class_balance=content['class_balance'],
                        data_balance=content['data_balance'],
                        epoch_padding=content['epoch_padding'],
                        data_padding=content['data_padding'],
                        task=content['task'])
        sample.set_data_padding(content['data_padding'],
                                max_len= None if content['max_len'] == \
                                    'Not Set' else content['max_len'])
        sample.set_x(content['x'])
        sample.set_y(content['y'])
        return sample

    def save(self, fpath):
        if self.__autoencoder:
            task = 'autoencoder'
        else:
            task = self.task
        content = {'unit' : self.get_unit(),
                   'tmin' : self.get_tmin(),
                   'tmax' : self.get_tmax(),
                   'n_splits' : self.get_n_splits(),
                   'test_size' : self.get_test_size(),
                   'class_balance' : self.get_class_balance(),
                   'data_balance' : self.get_data_balance(),
                   'epoch_padding' : self.epoch_padding,
                   'data_padding' : self.data_padding,
                   'max_len' : self.max_len if self.max_len is not None else \
                    'Not Set',
                   'x' : self.get_x(),
                   'y' : self.get_y(),
                   'task' : task}
        with open(fpath, 'w') as f:
            json.dump(content, f)

    def __check_editable(self):
        if not self.__editable:
            raise AssertionError('You cannot edit parameters of this object '
                                 'after specifying dataset. In another words, '
                                 'any change after using `Sample.from_dataset`'
                                 ' is not allowed.')

    def __check_mode(self, current_mode):
        if current_mode != self.mode:
            if current_mode == 'train':
                raise ModeError('You cannot use `train_set()` or `test_set()` '
                                'in `predict` mode. Please try `Sample.sample`'
                                '.')
            elif current_mode == 'predict':
                raise ModeError('You cannot use `sample()` in `train` mode. '
                                'Please try `Sample.train_set()` and '
                                '`Sample.test_set()` to sample them '
                                'respectively.')

    def __check_preparation(self):
        # 检查subgroup和split有没有做
        if not self.__preparation:
            self.init()

    def __check_labeldict(self, dataset):
        if not self.__autoencoder:
            for x in self.get_x():
                if dataset.elements[x] == 'label' and x not in \
                    dataset.labeldict:
                    raise LackOfLabelDictError('The element `{0}` does not '
                                               'have a labeldict. Please use '
                                               '`Dataset.one_hot()` function '
                                               'to create one or manually set '
                                               'one.'.format(x))
            for y in self.get_y():
                if y not in dataset.labeldict:
                    raise LackOfLabelDictError('The element `{0}` does not '
                                               'have a labeldict. Please use '
                                               '`Dataset.one_hot()` function '
                                               'to create one or manually set '
                                               'one.'.format(y))

    def __check_iteration(self):
        '''
        In `train` mode, before use `Sample.train_set()` or `Sample.test_set()`
        , we should get one fold first (or a subset for cross validation).
        Unless the cross validation is not needed (when `n_splits == 1`).
        
        If `n_splits != 1`, you should use `Sample.next_fold()` every iteration
        to initialize train_set and test_set.

        Raises
        -------
        KFoldError
            Raised when `Sample.next_fold()` does not be used properly.

        '''
        # look up if all the thing needed exist
        # that is, `self.train` & `self.test`
        if (not hasattr(self, 'train') or not hasattr(self, 'test')) and \
            self.n_splits > 1:
            raise KFoldError('you\'ve set `n_splits = {0}`, so you have {0}'
                             ' slices of `train_set` and `test_set`. You must '
                             'set an iteration by using `Sample.next_fold()`'
                             ' to initialize the train_set and test_set for '
                             'this slice.'.format(self.get_n_splits()))
        if (not hasattr(self, 'train') or not hasattr(self, 'test')) and \
            self.n_splits == 1:
            self.next_fold()

    def __check_dataset(self, dataset):
        # check `x`
        if self.get_x() is None:
            raise LackOfParameterError('`x` hasn\'t set. Please use `set_x` '
                                       'first.')
        else:
            for item in self.get_x():
                # check if `x` in dataset.elements
                if item not in dataset.elements.keys():
                    raise AssertionError('The target dataset has no element `'+
                                         item + '`.')
                # while unit == `epoch`, `x` cannot be `condition` (because
                # `condition` is not epoched feature, cannot represent an epoch)
                elif self.get_unit() == 'epoch' and \
                    dataset.elements[item] == 'condition':
                    raise AssertionError('while unit == `epoch`, `x` '
                                         'cannot be `condition` (because '
                                         '`condition` is not epoched '
                                         'feature, cannot represent an '
                                         'epoch)')
        # check `y`
        if self.mode == 'train':  # in training mode, `y` must provided
            if self.get_y() is None:
                raise LackOfParameterError('`y` hasn\'t set. '
                                           'Please use `set_y` first.')
            else:
                for item in self.get_y():
                    # check if `y` in dataset.elements
                    if item not in dataset.elements.keys():
                        raise AssertionError('The target dataset has no'
                                             ' element `' + item + '`.')
                    # check if `y` is `label` or `condition`
                    elif dataset.elements[item] != 'label' and \
                        dataset.elements[item] != 'condition':
                        if len(self.get_x()) == 1 and self.get_x() == \
                            self.get_y():
                            logging.info('Use mode AUTOENCODER.')
                            print('[INFO] Use mode AUTOENCODER.')
                            self.set_task('autoencoder')
                        else:
                            raise AssertionError('The `y` must be `label` or'
                                                 ' `condition`. But ' + item +
                                                 ' was a `' +
                                                 dataset.elements[item] +'`.')
                    # while unit == `data`, `y` must be `condition` (because
                    # `label` is epoched label, cannot represent a data)
                    elif self.get_unit() == 'data' and \
                        dataset.elements[item] == 'label':
                            raise AssertionError('while unit == `data`, `y` '
                                                 'must be `condition` (because'
                                                 ' `label` is epoched label, '
                                                 'cannot represent a data)')
        # and in prediction mode, `y` will not be used

    def from_dataset(self, dataset, data_selection:dict=None,
                     mode:str='train'):
        '''
        Bind your sampling plan with a `Dataset`, which enables you begin to
        sample, but also disallows you to change your sampling plan again.

        Parameters
        ----------
        dataset : sleepology.dataset.Dataset
            A `Dataset` object. You should first create a `Dataset` from source
            or load a `Dataset` from disk, then input the `Dataset` as a
            parameter.

        data_selection : dict, optional
            The data or epochs you want to sample. The default is None which
            means use all the data or epochs.
            You should input a dict whose keys are `element_name` and values
            are `element_value`.
            For example, you have a `Dataset`, it will be sampled in unit of
            epochs. And it has `PSG` as an element of feature, and `LABEL` as
            an element of label, and also `DIAGNOSE` as an element of
            condition. You just want to sample epochs whose `LABEL` is `1` or 
            '2' and `DIAGNOSE` is `'healthy'`, you can use:
            >>> data_selection = {'LABEL' : [1, 2].
            ...                   'DIAGNOSE' : ['healthy']}

        mode : {'train', 'predict'}, optional
            Define the purpose why you sample. If you want to train a model,
            the sampled data will be balanced and split into train set and
            test set. Else you want to use an existed model to predict new
            data, the `Sample` process should not do any change to data, and it
            will definitely no `Sample.y()` because the correct answer has not
            provided. The default is 'train'.

        Raises
        ------
        ValueError

        Returns
        -------
        None.

        '''
        # set mode
        try:
            if mode.lower() == 'train':
                self.mode = 'train'
            elif mode.lower() == 'predict':
                self.mode = 'predict'
                self.y = None
                if self.data_padding and self.max_len is None:
                    raise ValueError('In mode `predict`, max_len of '
                                     'data_padding must be set.')
            else:
                raise ValueError('Invalid mode. Must be \'train\' or '
                                 '\'predict\'.')

            # check dataset
            self.__check_dataset(dataset)
            self.__check_labeldict(dataset)
        except Exception as e:
            del self.mode
            raise e

        # set dataset
        self.dataset = dataset
        self.dataset.shape_check()
        # === SELECT DATA ===
        # # use all the data or epochs
        if data_selection is None:
            if self.get_unit() == 'epoch':
                self.__selection = None
                self.data_selection = self.dataset.select_epochs()
            elif self.get_unit() == 'data':
                self.__selection = None
                self.data_selection = self.dataset.select_data()
        # # select data
        else:
            # data_selection has parameter in form of dict
            self.__selection = data_selection
            self.data_selection = set()
            # case: epoch
            if self.get_unit() == 'epoch':
                for e in data_selection.keys():  # for each element
                    # [1] there is already a subset of epochs in the
                    # `self.data_selection` and the element `e` provides some
                    # selection range.
                    if self.data_selection and len(self.__selection[e]) != 0:
                        self.data_selection.intersection_update(
                            set(self.dataset.select_epochs(e, data_selection[e]))
                        )
                    # [2] `self.data_selection` is an empty set
                    elif (not self.data_selection) and \
                        len(data_selection[e]) != 0:
                        self.data_selection = \
                            set(self.dataset.select_epochs(e, data_selection[e]))
                    # [3] other situation
                    elif self.data_selection:
                        pass
                    else:
                        self.data_selection = \
                            self.dataset.select_epochs()
            # case: data
            elif self.get_unit() == 'data':
                for e in data_selection.keys():  # for each element
                    if self.data_selection and len(data_selection[e]) != 0:
                        self.data_selection.intersection_update(
                            set(self.dataset.select_data(e, data_selection[e]))
                        )
                    elif (not self.data_selection) and \
                        len(data_selection[e]) != 0:
                        self.data_selection = \
                            self.dataset.select_data(e, data_selection[e])
                    elif self.data_selection:
                        pass
                    else:
                        self.data_selection = \
                            self.dataset.select_data()
        # set max_len
        if self.data_padding and self.max_len is None:
            epochs = self.dataset.epochs_per_data(self.data_list())
            self.max_len = max(epochs)

        # print the summary
        self.summary()

        # lock the config
        self.__editable = False
        # init preparation
        self.__preparation = False

    def data_list(self):
        '''
        Get data list from all the epochs or data.

        Returns
        -------
        list
            The data list.

        '''
        rst = set()
        for item in self.data_selection:
            if isinstance(item, str):
                rst.add(item)
            else:
                rst.add(item[0])
        return list(rst)

    def init(self):
        '''
        Initialize all the steps before sampling, including padding and
        spliting the train_set and test_set.
        '''
        logging.debug('== CLEANING VARIABLES ==')
        if hasattr(self, 'train'):
            del self.train
        if hasattr(self, 'test'):
            del self.test
        if hasattr(self, 'onesamp'):
            del self.onesamp
        if hasattr(self, 'opc'):  # one per class
            del self.opc

        logging.info('== INITIALIZING SAMPLING ==')
        if self.task == 'classification':
            self.subgroups()
        self.shuffle_split()
        logging.info('== INITIALIZING SAMPLING ==')

    def subgroups(self):
        # Only used in classification tasks
        '''
        Split data or epochs by classes into subgroups.
        Generate `Sample.classes`, which is a `dict` of different classes.
        '''
                # this method only supports `classification` task, not `regression`
        # task. Here we check it.
        if self.task != 'classification':
            raise TaskError('Only `classification` task has subgroups. Please '
                            'check the returned table when doing '
                            '`Sample.from_dataset`.')
        logging.info('== DISCRIMINATE DIFFERENT CLASSES ==')
        if self.__selection:
            element = self.__selection
        else:
            element = self.get_y()
        classes = self.dataset.stat_classes(element, self.get_unit())
        self.classes = {}
        for c in classes:
            self.classes[c] = [i for i in classes[c] 
                               if i in self.data_selection]  # intersection of
                                                             # data_selection
                                                             # and classes
        logging.info('== DISCRIMINATE DIFFERENT CLASSES ==')

    def shuffle_split(self):
        '''
        Generate generators for each of classes.

        After using `Sample.shuffle_split`, subgroups will be split into 
        `Sample.get_n_splits()` pieces, and in each piece 
        `Sample.get_test_size()` of the data will act as test set.

        '''
        logging.info('== SHUFFLE AND SPLIT DATASET ==')
        logging.info('n_splits = {0}'.format(self.get_n_splits()))
        logging.info('test_size = {0}'.format(self.get_test_size()))
        ss = ShuffleSplit(n_splits=self.get_n_splits(),
                          test_size=self.get_test_size())
        self.__current_fold = iter(range(1, self.get_n_splits() + 1))
        if self.task == 'classification':
            self.__k_fold = {}
            for c in self.classes:
                self.__k_fold[c] = ss.split(self.classes[c])
        else:
            self.__k_fold = ss.split(self.data_selection)
        logging.info('== SHUFFLE AND SPLIT DATASET ==')

    def next_fold(self, shuffle=True, disturb=False):
        self.__check_preparation()
        # start a new iteration
        current_fold = next(self.__current_fold)
        logging.info('== GET FOLD {0} =='.format(current_fold))
        train_set = []
        test_set = []
        self.disturb = disturb
        debug_classes = 0
        debug_train_set_len = 0
        debug_test_set_len = 0
        if self.task == 'regression':
            train, test = next(self.__k_fold)
            train = [self.data_selection[i] for i in train]
            test = [self.data_selection[i] for i in test]
            train_set.extend(train)
            test_set.extend(test)
        else:
            # append train_set and test_set of each class
            for c in self.__k_fold:
                debug_classes += 1
                train, test = next(self.__k_fold[c])
                train = [self.classes[c][i] for i in train]
                test = [self.classes[c][i] for i in test]
                train_set.append(train)
                test_set.append(test)
                debug_train_set_len += len(train)
                debug_test_set_len += len(test)
            ## train_set is a 2-D list. The first dimension is `class`. The
            ## second dimension is `data` or `(data, epoch)` tuple.
            logging.info('Dataset has ' + str(debug_classes) + ' classes')
            logging.info('Current train_set length ' + str(debug_train_set_len))
            logging.info('Current test_set length ' + str(debug_test_set_len))
            if self.get_class_balance() and not self.get_data_balance():
                logging.info('Balancing classes ...')
                # find the max_len of different classes. Then oversample other
                # class to the max_len
                train_len = max(len(i) for i in train_set)
                logging.info('The max_len of train_set is ' + str(train_len))
                test_len = max(len(i) for i in test_set)
                logging.info('The max_len of test_set is ' + str(test_len))
                # balance every class
                for pcs in range(len(train_set)):  # each piece is a class
                    idx = np.random.choice(range(len(train_set[pcs])), train_len)
                    train_set[pcs] = [train_set[pcs][i] for i in idx]
                    logging.debug('[{0}] sample train_set to {1}.'.format(str(pcs),
                        str(len(train_set[pcs]))))
                for pcs in range(len(test_set)):  # each piece is a class
                    idx = np.random.choice(range(len(test_set[pcs])), test_len)
                    test_set[pcs] = [test_set[pcs][i] for i in idx]
                    logging.debug('[{0}] sample test_set to {1}.'.format(str(pcs),
                        str(len(test_set[pcs]))))
                train_set = flatten(train_set)
                test_set = flatten(test_set)
            elif self.get_data_balance() and not self.get_class_balance():
                logging.info('Balancing data ...')
                # this condition assert self.unit == 'epoch'
                # don't care about classes. Concatenate classes
                train_set = flatten(train_set)
                logging.info('Flatten train_set, length: ' + len(train_set))
                test_set = flatten(test_set)
                logging.info('Flatten test_set, length: ' + len(test_set))
                # and discriminate train_set and test_set by `data_name`
                train_set = list(self.__discriminate_data(train_set).values())
                test_set = list(self.__discriminate_data(test_set).values())
                ## Now, the train_set is also a 2-D list. The first dimension is
                ## `data`. The second dimension is `(data, epoch)` tuple.
                # find the max_len of different data. Then oversample other data
                # to the max_len
                train_len = max(len(i) for i in train_set)
                logging.info('The max_len of train_set is ' + str(train_len))
                test_len = max(len(i) for i in test_set)
                logging.info('The max_len of test_set is ' + str(test_len))
                # balance every data
                for pcs in range(len(train_set)):  # each piece is a datum
                    idx = np.random.choice(range(len(train_set[pcs])), train_len)
                    train_set[pcs] = [train_set[pcs][i] for i in idx]
                    logging.debug('[{0}] sample train_set to {1}.'.format(str(pcs),
                        str(len(train_set[pcs]))))
                for pcs in range(len(test_set)):  # each piece is a datum
                    idx = np.random.choice(range(len(test_set[pcs])), test_len)
                    test_set[pcs] = [test_set[pcs][i] for i in idx]
                    logging.debug('[{0}] sample test_set to {1}.'.format(str(pcs),
                        str(len(test_set[pcs]))))
                train_set = flatten(train_set)
                test_set = flatten(test_set)
            elif self.get_class_balance() and self.get_data_balance():
                logging.info('Balancing data and balancing classes ...')
                logging.info('> Discriminate different data in classes. So it '
                             'generate a (class, data) structure')
                tem = []
                for ts in train_set:
                    ## extending a 2-D list. The first dimension is `data`. The
                    ## second dimension is `(data, epoch)` tuple.
                    tem.extend(list(self.__discriminate_data(ts).values()))
                train_set = tem
                tem = []
                for tt in test_set:
                    ## extending a 2-D list. The first dimension is `data`. The
                    ## second dimension is `(data, epoch)` tuple.
                    tem.extend(list(self.__discriminate_data(tt).values()))
                test_set = tem
                ## the train_set and test_set remain a 2-D array because we have
                ## used `extend` instead of `append`. The first dimension is a
                ## combination of `data` and `class`. A list of certain
                ## `(data, class)` in the second dimension.
                # compute max_len of train_set and test_set
                train_len = max(len(i) for i in train_set)
                logging.info('The max_len of train_set is ' + str(train_len))
                test_len = max(len(i) for i in test_set)
                logging.info('The max_len of test_set is ' + str(test_len))
                # balance every (data, class)
                for pcs in range(len(train_set)):
                    idx = np.random.choice(range(len(train_set[pcs])), train_len)
                    train_set[pcs] = [train_set[pcs][i] for i in idx]
                    logging.debug('[{0}] sample train_set to {1}.'.format(str(pcs),
                        str(len(train_set[pcs]))))
                for pcs in range(len(test_set)):
                    idx = np.random.choice(range(len(test_set[pcs])), test_len)
                    test_set[pcs] = [test_set[pcs][i] for i in idx]
                    logging.debug('[{0}] sample test_set to {1}.'.format(str(pcs),
                        str(len(test_set[pcs]))))
                train_set = flatten(train_set)
                test_set = flatten(test_set)
            else:
                train_set = flatten(train_set)
                test_set = flatten(test_set)
        # finally
        if shuffle:
            logging.info('shuffle...')
            random.shuffle(train_set)
            random.shuffle(test_set)
        self.train = train_set
        self.test = test_set
        # disturb, which disturb the match of x and y, i.e. shuffle the y
        if disturb:
            train_set = train_set.copy()
            test_set = test_set.copy()
            random.shuffle(train_set)
            random.shuffle(test_set)
            self.train_y = train_set
            self.test_y = test_set
        logging.info('Length of `train_set` this fold: {0}'.format(
            len(self.train)))
        logging.info('Length of `test_set` this fold: {0}'.format(
            len(self.test)))
        logging.info('== GET FOLD {0} =='.format(current_fold))
        return current_fold

    def __discriminate_data(self, ori):
        '''
        Discriminate different data in a series of epochs.
        
        The input is a list whose element has a form `(data_name, epoch)`.
        The output is dict:
            output = {data_name: [(data_name, epoch), (data_name, epoch), ...],
                      data_name: [(data_name, epoch), (data_name, epoch), ...],
                      ...}
        Example:
        >>> list(Sample.__discriminate_data(ori).keys())  # acquire a data list
        
        >>> list(Sample.__discriminate_data(ori).values())  # split different
                                                            # data

        Parameters
        ----------
        ori : list
            A list whose element has a form `(data_name, epoch)`.

        Returns
        -------
        rst : dict
            A dict, in form of:
            output = {data_name: [(data_name, epoch), (data_name, epoch), ...],
                      data_name: [(data_name, epoch), (data_name, epoch), ...],
                      ...}

        '''
        rst = {}
        for t in ori:
            if t[0] not in rst:
                rst[t[0]] = []
            rst[t[0]].append(t)
        return rst

    def train_set(self, generator=True):
        if generator:
            logging.info('Use generator')
            return self.train_set_generator()
        else:
            logging.info('Do not use generator')
            x_samp = []
            y_samp = []
            for x, y in self.train_set_generator():
                x_samp.append(x)
                y_samp.append(y)
            x_samp = np.asarray(x_samp)
            y_samp = np.asarray(y_samp)
            return (x_samp, y_samp)

    def train_set_generator(self):
        self.__check_mode('train')
        self.__check_iteration()
        for idx, item in enumerate(self.train):
            try:
                if self.get_unit() == 'epoch' and not self.disturb:
                    yield self.dataset.sample_epoch(
                        item[0],
                        item[1],
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(), 
                        epoch_padding=self.epoch_padding,
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
                elif self.get_unit() == 'data' and not self.disturb:
                    yield self.dataset.sample_data(
                        item,
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(),
                        data_padding=self.data_padding,
                        max_len=self.max_len,
                        epoch_padding=self.epoch_padding,
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
                elif self.get_unit() == 'epoch' and self.disturb:
                    yield self.dataset.sample_epoch(
                        item[0],
                        item[1],
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(), 
                        epoch_padding=self.epoch_padding,
                        test_data_name=self.train_y[idx][0],
                        test_epoch=self.train_y[idx][1],
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
                elif self.get_unit() == 'data' and not self.disturb:
                    yield self.dataset.sample_data(
                        item,
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(),
                        data_padding=self.data_padding,
                        max_len=self.max_len,
                        epoch_padding=self.epoch_padding,
                        test_data_name=self.train_y[idx],
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
            except BrokenTimestepError:
                continue

    def test_set(self, generator=True):
        if generator:
            logging.info('Use generator')
            return self.test_set_generator()
        else:
            logging.info('Do not use generator')
            x_samp = []
            y_samp = []
            for x, y in self.test_set_generator():
                x_samp.append(x)
                y_samp.append(y)
            x_samp = np.asarray(x_samp)
            y_samp = np.asarray(y_samp)
            return (x_samp, y_samp)

    def test_set_generator(self):
        self.__check_mode('train')
        self.__check_iteration()
        for idx, item in enumerate(self.test):
            try:
                if self.get_unit() == 'epoch' and not self.disturb:
                    yield self.dataset.sample_epoch(
                        item[0],
                        item[1],
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(), 
                        epoch_padding=self.epoch_padding,
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
                elif self.get_unit() == 'data' and not self.disturb:
                    yield self.dataset.sample_data(
                        item,
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(),
                        data_padding=self.data_padding,
                        max_len=self.max_len,
                        epoch_padding=self.epoch_padding,
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
                elif self.get_unit() == 'epoch' and self.disturb:
                    yield self.dataset.sample_epoch(
                        item[0],
                        item[1],
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(), 
                        epoch_padding=self.epoch_padding,
                        test_data_name=self.test_y[idx][0],
                        test_epoch=self.test_y[idx][1],
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
                elif self.get_unit() == 'data' and self.disturb:
                    yield self.dataset.sample_data(
                        item,
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(),
                        data_padding=self.data_padding,
                        max_len=self.max_len,
                        epoch_padding=self.epoch_padding,
                        test_data_name=self.test_y[idx],
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
            except BrokenTimestepError:
                continue

    def one(self, generator:bool=True):
        '''
        Sample only one data and its label in dataset. This one data will
        be repeatedly sampled and fed into the model. This method is designed
        for testing a model. If the model cannot fit just one sample, there
        maybe some problems with the model, such as gradient disappearance.

        Parameters
        ----------
        generator : bool, optional
            Determine the return type. If True, the returned object is a
            generator, it is supported by tensorflow and memory-saving. If
            False, the returned object is a np.ndarray, it is supported by
            scikit-learn.
            The default is True.

        Returns
        -------
        generator or np.ndarray

        '''
        if generator:
            logging.info('Use generator')
            return self.one_generator()
        else:
            logging.info('Do not use generator')
            x_samp = []
            y_samp = []
            for x, y in self.one_generator():
                x_samp.append(x)
                y_samp.append(y)
            x_samp = np.asarray(x_samp)
            y_samp = np.asarray(y_samp)
            return (x_samp, y_samp)

    def one_generator(self):
        '''
        Sample only one data and its label in dataset. This one data will
        be repeatedly sampled and fed into the model. This method is designed
        for testing a model. If the model cannot fit just one sample, there
        maybe some problems with the model, such as gradient disappearance.

        Yields
        ------
        np.ndarray

        '''
        # the purpose of `Sample.one` is only to get one sample, regardless of
        # whether the mode `train` or `test`, or whether the dataset needs to
        # cross validation or not.
        if not hasattr(self, 'onesamp'):
            onesamp = random.sample(self.data_selection, 1)
            self.onesamp = onesamp * len(self.data_selection)
        for idx, item in enumerate(self.onesamp):
            try:
                if self.get_unit() == 'epoch':
                    yield self.dataset.sample_epoch(
                        item[0],
                        item[1],
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(), 
                        epoch_padding=self.epoch_padding,
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
                elif self.get_unit() == 'data':
                    yield self.dataset.sample_data(
                        item,
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(),
                        data_padding=self.data_padding,
                        max_len=self.max_len,
                        epoch_padding=self.epoch_padding,
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
            except BrokenTimestepError:
                continue
    
    def one_per_class(self, generator=True):
        '''
        Sample only one data and its label per class. These data will be
        repeatedly sampled and fed into the model. This method is designed
        for testing a model. If the model cannot fit such small dataset, there
        maybe some problems with the model, such as gradient disappearance.

        Parameters
        ----------
        generator : bool, optional
            Determine the return type. If True, the returned object is a
            generator, it is supported by tensorflow and memory-saving. If
            False, the returned object is a np.ndarray, it is supported by
            scikit-learn.
            The default is True.

        Returns
        -------
        generator or np.ndarray

        '''
        # this method only supports `classification` task, not `regression`
        # task. Here we check it.
        if self.task != 'classification':
            raise TaskError('Only `classification` task has classes. Please '
                            'check the returned table when doing '
                            '`Sample.from_dataset` if your task is '
                            'classification, or use `Sample.one()` instead '
                            'of `Sample.one_per_class()` in regression task,')
        if generator:
            logging.info('Use generator')
            return self.one_per_class_generator()
        else:
            logging.info('Do not use generator')
            x_samp = []
            y_samp = []
            for x, y in self.one_per_class_generator():
                x_samp.append(x)
                y_samp.append(y)
            x_samp = np.asarray(x_samp)
            y_samp = np.asarray(y_samp)
            return (x_samp, y_samp)

    def one_per_class_generator(self):
        '''
        Sample only one data and its label per class. These data will be
        repeatedly sampled and fed into the model. This method is designed
        for testing a model. If the model cannot fit such small dataset, there
        maybe some problems with the model, such as gradient disappearance.

        Yields
        ------
        np.ndarray

        '''
        if not hasattr(self, 'opc'):
            if not hasattr(self, 'classes'):
                self.subgroups()
            # the purpose of `Sample.one_per_class` is only to get samples,
            # regardless of whether the mode `train` or `test`, or whether the
            # dataset needs to cross validation or not.
            opc = [] # one_per_class
            for k in self.classes.keys():
                opc.extend(random.sample(self.classes[k], 1))
            # the length of samples = data_length
            self.opc = opc * math.floor(
                len(self.data_selection) / len(self.classes.keys()))
        for idx, item in enumerate(opc):
            try:
                if self.get_unit() == 'epoch':
                    yield self.dataset.sample_epoch(
                        item[0],
                        item[1],
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(), 
                        epoch_padding=self.epoch_padding,
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
                elif self.get_unit() == 'data':
                    yield self.dataset.sample_data(
                        item,
                        (self.get_x(), self.get_y()),
                        tmin=self.get_tmin(),
                        tmax=self.get_tmax(),
                        data_padding=self.data_padding,
                        max_len=self.max_len,
                        epoch_padding=self.epoch_padding,
                        autoencoder=self.__autoencoder,
                        array_type=self.array_type,
                        concat=self.concat,
                        x_dict = self.x_dict,
                        y_dict = self.y_dict)
            except BrokenTimestepError:
                continue

    def sample(self):
        self.__check_mode('predict')
        pass
    