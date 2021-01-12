# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:31:23 2020

@author: Zhao Kuangshi
"""
from .source import *
from .utils import total_size
# from .procedure import Procedure
from .labeldict import BaseDict, AASM, ClassDict
from .exception_logger import ExceptionLogger
from .exceptions import DataStateError, BrokenTimestepError

import h5py
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import logging
import traceback
from typing import Union, Dict


os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

package_root = os.path.dirname(os.path.abspath(__file__))

class Dataset(object):
    '''
    2020-2-20
    引入数据、加载数据、预处理数据、记录日志、*合并标签、*输出文件、*输出数组。
    '''
    # the state code
    ADDED = 0
    CHECKED = 1
    PREPROCESSED = 2
    PREDICTED = 3
    ERROR = -1
    
    def __init__(self, dataset_name, save_path, comment = '', 
                labeldict: Union[BaseDict, Dict[str, BaseDict], None] = None,
                mode = 'memory'):
        '''
        Create a new `Dataset()`

        Parameters
        ----------
        dataset_name : str
            The name of the dataset. It will be the filename if you save this
            dataset.
        save_path : path-like
            Where you want to save this dataset.
        comment : str, optional
            A comment string to describe your dataset. The default is ''.
        labeldict : instance of LabelDict OR a dictionary of LabelDict whose
            keys are label names, optional

            A dictionary to manage labels of dataset. It will translate human-
            readable label to the style which fits machine learning.
        mode : {'memory', 'disk'}, optional
            For the faster speed, 'memory' mode will process all the data in
            the memory. But if you have a huge dataset which exceeds or will
            exhaust your memory, the 'disk' mode will cache all the data at
            disk until you are to use them. It do slow down the computation,
            but it is a good resolution when you are not able to extend your
            memory. The default is 'memory'.

        '''
        self.VERSION = 0.2
        self.path = save_path
        if self.path is not None and (dataset_name is not None):
            self.set_name(dataset_name)
        self.comment = comment
        # set LabelDict
        self.labeldict = {}
        if isinstance(labeldict, BaseDict):
            self.set_labeldict('LABEL', labeldict)
        elif isinstance(labeldict, dict):
            for label in labeldict.keys():
                self.set_labeldict(label, labeldict[label])
        # set mode
        if mode.lower() == 'memory':
            self.disk_mode = False
        else:
            self.disk_mode = True
        self.__init_dataset()
        self.exception_logger = ExceptionLogger(self)
        self.elements = dict()
        self.features = set()
        self.labels = set()
        self.has_label = False

    def disk_file(self):
        if not hasattr(self, 'df'): # The cache not being created
            cache_dir = os.path.expanduser(os.path.join('~', '.sleepology'))
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            tf = tempfile.TemporaryFile(dir = cache_dir)
            self.df = h5py.File(tf, 'w')
            return self.df
        else:
            return self.df

    def set_labeldict(self, label_name, labeldict):
        self.labeldict[label_name] = labeldict

    ### NAME ###

    def set_name(self, name):
        '''
        Set the name of the dataset.

        Parameters
        ----------
        name : str
            The name of the dataset.

        '''
        # 2020-2-23 改名字同时改保存路径
        self.name = name
        self.save_direction = os.path.join(os.path.abspath(self.path), 
                                           self.name + '.dataset')

    def get_name(self):
        '''
        Get the name of the dataset.

        Returns
        -------
        str
            The name of the dataset.

        '''
        return self.name

    ### COMMENT ###

    def set_comment(self, comment):
        '''
        Set the comment of the dataset. The comment is used to discriminate 
        datasets.

        Parameters
        ----------
        comment : str
            The comment.

        '''
        self.comment = comment

    def get_comment(self):
        '''
        Get the comment of the dataset. The comment is used to discriminate 
        datasets.

        Returns
        -------
        str
            The comment.

        '''
        return self.comment

    ### DATASET ###

    def __init_dataset(self):
        if self.disk_mode:
            self.__df_init_dataset(self.disk_file())
        else:
            self.__mm_init_dataset()

    def __mm_init_dataset(self):
        self.dataset = {}

    def __df_init_dataset(self, disk_file):
        disk_file.create_group('dataset')

    ### ELEMENT ###

    def __new_feature(self, feature_name):
        if feature_name in self.elements and self.elements[feature_name] != 'feature':
            raise ValueError('Feature name conflict. A `' +
                             self.elements[feature_name] + '` has same name.')
        elif feature_name in self.elements and self.elements[feature_name] == 'feature':
            pass
        else:
            self.features.add(feature_name)
            self.elements[feature_name] = 'feature'

    def __new_label(self, label_name):
        if label_name in self.elements and self.elements[label_name] != 'label':
            raise ValueError('Label name conflict. A `' +
                             self.elements[label_name] + '` has same name.')
        elif label_name in self.elements and self.elements[label_name] == 'label':
            pass
        else:
            self.labels.add(label_name)
            self.elements[label_name] = 'label'

    def __new_condition(self, condition_name):
        if condition_name in self.elements and self.elements[condition_name] != 'condition':
            raise ValueError('Condition name conflict. A `' +
                             self.elements[condition_name] + '` has same name.')
        if condition_name in self.elements and self.elements[condition_name] == 'condition':
            pass
        else:
            self.elements[condition_name] = 'condition'

    ### DATA ###

    def __init_new_data(self, data_name):
        if self.disk_mode:
            self.__df_init_new_data(self.disk_file(), data_name)
        else:
            self.__mm_init_new_data(data_name)

    def __mm_init_new_data(self, data_name):
        # initialize the new data
        self.dataset[data_name] = dict()
        self.dataset[data_name]['source'] = dict()
        self.dataset[data_name]['data'] = dict()

    def __df_init_new_data(self, disk_file, data_name):
        disk_file['dataset'].create_group(data_name)
        disk_file['dataset'][data_name].create_group('data')

    def add_data(self, feature_source, label_source = None, data_name = None):
        # format the source input
        if isinstance(feature_source, Source):
            feature_source = {'FEATURE': feature_source}
        if isinstance(label_source, Source):
            label_source = {'LABEL': label_source}
        # ensure the `data_name`. If the same name has existed, append
        # number behind
        main_source = list(feature_source.keys())[0]
        if data_name is None:
            data_name = feature_source[main_source].name
        if data_name in self.get_data() and (self.get_source(data_name,\
            main_source).path == feature_source[main_source].path): 
            pass
        else: 
            if data_name in self.get_data(): 
                i = 1
                while (data_name + '_' + str(i)) in self.get_data():
                    i += 1
                data_name = (data_name + '_' + str(i))

            self.__init_new_data(data_name)
            
            # register the source
            for feature_name in feature_source.keys():
                self.__new_feature(feature_name)
                self.__set_source(data_name, feature_name, 
                                  feature_source[feature_name]) 
            if label_source is not None:
                self.has_label = True
                for label_name in label_source.keys():
                    self.__new_label(label_name)
                    self.__set_source(data_name, label_name,
                                      label_source[label_name])
            
            self.__set_state(data_name, Dataset.ADDED)

    def del_data(self, data_name = None):
        '''
        Delete data from the dataset.

        Parameters
        ----------
        data_name : str, optional
            The data to be deleted. If None, all the data will be deleted.

        '''
        if self.disk_mode:
            self.__df_del_data(self.disk_file(), data_name)
        else:
            self.__mm_del_data(data_name)

    def __mm_del_data(self, data_name = None):
        '''
        Delete data from the dataset.

        Parameters
        ----------
        data_name : str, optional
            The data to be deleted. If None, all the data will be deleted.

        '''
        if data_name is None:
            self.dataset = {}
        else:
            self.dataset.pop(data_name)

    def __df_del_data(self, disk_file, data_name = None):
        if data_name is None:
            disk_file['dataset'].clear()
        else:
            disk_file['dataset'].pop(data_name)

    def get_data(self, data_name = None, attr = None):
        if self.disk_mode:
            return self.__df_get_data(self.disk_file(), data_name, attr)
        else:
            return self.__mm_get_data(data_name, attr)

    def __mm_get_data(self, data_name = None, attr = None):
        # 2020-3-1
        if data_name is None:
            return self.dataset.keys()
        elif attr is None:
            return self.dataset.get(data_name)
        else:
            return self.dataset.get(data_name).get(attr)

    def __df_get_data(self, disk_file, data_name = None, attr = None):
        if data_name is None:
            return disk_file['dataset'].keys()
        elif attr is None:
            return disk_file['dataset'][data_name]
        elif attr in disk_file['dataset'][data_name].attrs:
            return disk_file['dataset'][data_name].attrs[attr]
        else:
            return disk_file['dataset'][data_name][attr]

    def select_data(self, condition_type = None, condition_value = None):
        # 删除所有文件中指定label的数据
        if condition_type is None and condition_value is None:
            rst = set()
            for data_name in self.get_data():
                if self.__get_state(data_name) < Dataset.PREPROCESSED:
                    continue
                logging.info(data_name)
                rst.add(data_name)
        elif condition_type is not None and condition_value is not None:
            rst = set()
            if not isinstance(condition_value, list):
                condition_value = [condition_value]
            for l in condition_value:
                for data_name in self.get_data():
                    if self.__get_state(data_name) < Dataset.PREPROCESSED:
                        continue
                    logging.info(data_name)
                    if self.get_condition(data_name, condition_type) == l:
                        rst.add(data_name)
        else:
            raise TypeError('You should input all of or none of\n' + 
                            '`condition_type` and `condition_value`')
        return rst

    def exclude_data(self, condition_type, condition_value):
        if not isinstance(condition_value, list):
            condition_value = [condition_value]
        for l in condition_value:
            to_be_del = []
            for data_name in self.get_data():
                if self.__get_state(data_name) != Dataset.ERROR:
                    logging.info(data_name)
                    if self.get_condition(data_name, condition_type) == l:
                        to_be_del.append(data_name)
            for d in to_be_del:
                self.del_data(d)

    def epochs_per_data(self, data_name = None):
        r = []
        if data_name is None:
            data = self.get_data()
        elif isinstance(data_name, str):
            data = [data_name]
        elif isinstance(data_name, list):
            data = data_name
        for d in data:
            if self.__get_state(d) == Dataset.PREPROCESSED:
                r.append(len(self.get_epochs(d)))
        return r

    ### CONDITION ###
    def has_condition(self, data_name, condition_type = None):
        if self.disk_mode:
            return self.__df_has_condition(self.disk_file(),
                                           data_name, condition_type)
        else:
            return self.__mm_has_condition(data_name, condition_type)

    def __mm_has_condition(self, data_name, condition_type = None):
        if condition_type is None:
            return 'CONDITION' in self.dataset[data_name]
        elif 'CONDITION' in self.dataset[data_name]:
            return condition_type in self.dataset[data_name]['CONDITION']
        else:
            return False

    def __df_has_condition(self, disk_file, data_name, condition_type = None):
        if condition_type is None:
            return 'CONDITION' in disk_file['dataset'][data_name].keys()
        elif 'CONDITION' in disk_file['dataset'][data_name]:
            return condition_type in disk_file['dataset'][data_name]\
                ['CONDITION'].attrs
        else:
            return False

    def set_condition(self, data_name, condition_type, condition_label):
        if self.disk_mode:
            self.__df_set_condition(self.disk_file(), data_name,
                                    condition_type, condition_label)
        else:
            self.__mm_set_condition(data_name, condition_type, condition_label)

    def __mm_set_condition(self, data_name, condition_type, condition_label):
        if not self.has_condition(data_name):
            self.dataset[data_name]['CONDITION'] = {}
        self.__new_condition(condition_type)
        self.dataset[data_name]['CONDITION'][condition_type] = condition_label

    def __df_set_condition(self, disk_file, data_name, condition_type,
                           condition_label):
        if not self.__df_has_condition(disk_file, data_name):
            disk_file['dataset'][data_name].create_group('CONDITION')
        self.__new_condition(condition_type)
        disk_file['dataset'][data_name]['CONDITION'].\
            attrs[condition_type] = condition_label

    def get_condition(self, data_name, condition_type = None):
        if self.disk_mode:
            return self.__df_get_condition(self.disk_file(), data_name,
                                           condition_type)
        else:
            return self.__mm_get_condition(data_name, condition_type)

    def __mm_get_condition(self, data_name, condition_type = None):
        if condition_type is None and self.__mm_has_condition(data_name,
                                                              condition_type):
            return self.dataset[data_name]['CONDITION']
        elif self.has_condition(data_name, condition_type):
            return self.dataset[data_name]['CONDITION'][condition_type]
        else:
            return False

    def __df_get_condition(self, disk_file, data_name, condition_type = None):
        if condition_type is None and \
            self.__df_has_condition(disk_file, data_name, condition_type):
            r = {}
            for condition_type in disk_file['dataset']\
                [data_name]['CONDITION'].attrs:
                r[condition_type] = disk_file['dataset']\
                    [data_name]['CONDITION'].attrs[condition_type]
            return r
        elif self.__df_has_condition(disk_file, data_name, condition_type):
            return disk_file['dataset'][data_name]['CONDITION'].\
                attrs[condition_type]
        else:
            return False

    def del_condition(self, data_name, condition_type = None):
        if self.disk_mode:
            self.__df_del_condition(self.disk_file(), data_name, condition_type)
        else:
            self.__mm_del_condition(data_name, condition_type)

    def __mm_del_condition(self, data_name, condition_type = None):
        if condition_type is None and self.has_condition(data_name):
            self.dataset[data_name].pop('CONDITION')
        elif condition_type is not None and self.has_condition(data_name, condition_type):
            self.dataset[data_name]['CONDITION'].pop(condition_type)

    def __df_del_condition(self, disk_file, data_name, condition_type = None):
        if condition_type is None and self.__df_has_condition(data_name):
            disk_file['dataset'][data_name].pop('CONDITION')
        elif condition_type is not None and \
            self.__df_has_condition(data_name, condition_type):
            disk_file['dataset'][data_name]['CONDITION'].\
                attrs.pop(condition_type)

    ### EPOCH ###

    def __get_epoch(self, data_name, epoch = None):
        if self.disk_mode:
            return self.__df_get_epoch(self.disk_file(), data_name, epoch)
        else:
            return self.__mm_get_epoch(data_name, epoch)

    def __mm_get_epoch(self, data_name, epoch = None):
        if epoch is None:
            return self.__mm_get_epochs(data_name)
        else:
            return self.dataset[data_name]['data'][epoch]

    def __df_get_epoch(self, disk_file, data_name, epoch = None):
        if epoch is None:
            return self.__df_get_epochs(disk_file, data_name)
        else:
            # this returns a `h5py.Group` object
            return disk_file['dataset'][data_name]['data'][str(epoch)]

    def get_epochs(self, data_name):
        if self.disk_mode:
            return self.__df_get_epochs(self.disk_file(), data_name)
        else:
            return self.__mm_get_epochs(data_name)

    def __mm_get_epochs(self, data_name):
        return sorted(list(self.dataset[data_name]['data'].keys()))

    def __df_get_epochs(self, disk_file, data_name):
        return sorted(int(e) for e in 
                      list(disk_file['dataset'][data_name]['data'].keys()))

    def __init_epoch(self, data_name, epoch):
        if self.disk_mode:
            self.__df_init_epoch(self.disk_file(), data_name, epoch)
        else:
            self.__mm_init_epoch(data_name, epoch)

    def __mm_init_epoch(self, data_name, epoch):
        self.dataset[data_name]['data'][epoch] = dict()

    def __df_init_epoch(self, disk_file, data_name, epoch):
        disk_file['dataset'][data_name]['data'].create_group(str(epoch))

    def __del_epoch(self, data_name, epoch):
        if self.disk_mode:
            self.__df_del_epoch(self.disk_file(), data_name, epoch)
        else:
            self.__mm_del_epoch(data_name, epoch)

    def __mm_del_epoch(self, data_name, epoch):
        self.dataset[data_name]['data'].pop(epoch)

    def __df_del_epoch(self, disk_file, data_name, epoch):
        disk_file['dataset'][data_name]['data'].pop(str(epoch))

    def select_epochs(self, element_name=None, element_value=None,
                      data_name=None):
        # data_name
        if data_name is None:
            data_name = self.get_data()
        elif isinstance(data_name, str):
            data_name = [data_name]
        # label_name & label_value
        if element_name is None:
            rst = list()
            for d in data_name:
                if self.__get_state(d) < Dataset.PREPROCESSED:
                    continue
                logging.info(d)
                for epoch in self.get_epochs(d):
                    rst.append((d, epoch))
        elif element_name is not None and element_value is not None:
            rst = list()
            if not isinstance(element_value, list):
                element_value = [element_value]
            for d in data_name:
                if self.__get_state(d) == Dataset.ERROR:
                    continue
                logging.info(d)
                if self.elements[element_name] == 'label':
                    for epoch in self.get_epochs(d):
                        if self.get_label(d, epoch, element_name) in \
                            element_value:
                            rst.append((d, epoch))
                elif self.elements[element_name] == 'condition':
                    if self.get_condition(d, element_name) in element_value:
                        # if matches the condition, add all the epochs into rst
                        for epoch in self.get_epochs(d):
                            rst.append((d, epoch))
        else:
            raise TypeError('You should input all of or none of\n' + 
                            '`element_name` and `element_value`')
        return rst

    def exclude_epochs(self, label_name = None, label_value = None):
        # 删除所有文件中指定label的数据
        if (label_name is not None) and (label_value is not None):
            if not isinstance(label_value, list):
                label_value = [label_value]
            for l in label_value:
                for data_name in self.get_data():
                    if self.__get_state(data_name) == Dataset.ERROR:
                        continue
                    logging.info(data_name)
                    to_be_del = []
                    for epoch in self.get_epochs(data_name):
                        if self.get_label(data_name, epoch, label_name) == l:
                            to_be_del.append(epoch)
                    for e in to_be_del:
                        self.__del_epoch(data_name, e)
        else:
            # When no inputs, all the labels NOT IN LABELDICT will be exclude.
            for label_name in self.labels:
                allowed_label = self.labeldict[label_name].labels()
                for data_name in self.get_data():
                    if self.__get_state(data_name) == Dataset.ERROR:
                        continue
                    logging.info(data_name)
                    to_be_del = []
                    for epoch in self.get_epochs(data_name):
                        if self.get_label(data_name, epoch, label_name) \
                            not in allowed_label:
                            to_be_del.append(epoch)
                    for e in to_be_del:
                        self.__del_epoch(data_name, e)

    def __get_epoch_element_list(self, data_name, epoch):
        if self.disk_mode:
            return self.__df_get_epoch_element_list(self.disk_file(),
                                                    data_name, epoch)
        else:
            return self.__mm_get_epoch_element_list(data_name, epoch)
        
    def __mm_get_epoch_element_list(self, data_name, epoch):
        return list(self.dataset[data_name]['data'][epoch].keys())    

    def __df_get_epoch_element_list(self, disk_file, data_name, epoch):
        r = list(disk_file['dataset'][data_name]['data'][str(epoch)].keys())
        r.extend(list(disk_file['dataset'][data_name]\
                      ['data'][str(epoch)].attrs))
        return r

    ### FEATURE & LABEL ###

    def __set_feature(self, data_name, epoch, feature_name, value):
        '''
        A private method for setting features of dataset. The features are 
        NumPy matrices which represent the data. Most of datasets have \'single 
        feature matrix and single label\'. That means the ML models have single
        input and single output. However, `sleepology` also supports datasets 
        with multiple features and labels. Use different `feature_name` to 
        distinguish different feature matrices.
        
        NOTE: The feature matrices with the SAME feature_name in different 
        epochs or data MUST HAVE THE SAME SHAPE.

        Parameters
        ----------
        data_name : str
            The `data_name` of the data which should be in the 
            `self.dataset.keys()`.
        epoch : int
            The serial number of the epoch in PSG data.
        feature_name : str
            The name for distinguishing different feature matrices. It is 
            necessary in dataset with multiple features. The default is 
            \'FEATURE\'.
        value : np.ndarray or array-like
            The feature matrix.

        Returns
        -------
        None.

        '''
        if self.disk_mode:
            self.__df_set_feature(self.disk_file(), data_name,
                                  epoch, feature_name, value)
        else:
            self.__mm_set_feature(data_name, epoch, feature_name, value)

    def __mm_set_feature(self, data_name, epoch, feature_name, value):
        # initialize a new epoch when the feature value first being set.
        if epoch not in self.get_epochs(data_name):
            self.__init_epoch(data_name, epoch)
        
        self.dataset[data_name]['data'][epoch][feature_name] = value

    def __df_set_feature(self, disk_file, data_name,
                         epoch, feature_name, value):
        if epoch not in self.__df_get_epochs(disk_file, data_name):
            self.__df_init_epoch(disk_file, data_name, epoch)
        if feature_name in disk_file['dataset'][data_name]['data'][str(epoch)]:
            disk_file['dataset'][data_name]['data']\
                [str(epoch)][feature_name][()] = value
        else:
            disk_file['dataset'][data_name]['data'][str(epoch)].\
                create_dataset(feature_name, data = value)

    def get_feature(self, data_name, epoch, feature_name):
        '''
        A private matrix for getting feature matrix of particular data, epoch 
        and feature_name.

        Parameters
        ----------
        data_name : str
            The `data_name` of the data which should be in the 
            `self.dataset.keys()`.
        epoch : int
            The serial number of the epoch in PSG data.
        feature_name : str
            The name for distinguishing different feature matrices. It is 
            necessary in dataset with multiple features. The default is 
            \'FEATURE\'.

        Returns
        -------
        np.ndarray or array-like
            The feature matrix.

        '''
        if self.disk_mode:
            return self.__df_get_feature(self.disk_file(), data_name, 
                                         epoch, feature_name)
        else:
            return self.__mm_get_feature(data_name, epoch, feature_name)

    def __mm_get_feature(self, data_name, epoch, feature_name):
        return np.squeeze(self.dataset[data_name]['data'][epoch][feature_name])


    def __df_get_feature(self, disk_file, data_name, epoch, feature_name):
        rst =  disk_file['dataset'][data_name]['data']\
            [str(epoch)][feature_name][()]
        return np.squeeze(rst)

    def __set_label(self, data_name, epoch, label_name, value):
        '''
        A private method for setting labels of dataset. The labels are 
        NumPy matrices which represent the data. Most of datasets have \'single 
        feature matrix and single label\'. That means the ML models have single
        input and single output. However, `sleepology` also supports datasets 
        with multiple features and labels. Use different `label_name` to 
        distinguish different label matrices.
        
        NOTE: The label matrices with the SAME label_name in different 
        epochs or data MUST HAVE THE SAME SHAPE.

        Parameters
        ----------
        data_name : str
            The `data_name` of the data which should be in the 
            `self.dataset.keys()`.
        epoch : int
            The serial number of the epoch in PSG data.
        label_name : str
            The name for distinguishing different label matrices. It is 
            necessary in dataset with multiple labels. The default is 
            \'LABEL\'.
        value : np.ndarray or array-like
            The label matrix.

        Returns
        -------
        None.

        '''
        if self.disk_mode:
            self.__df_set_label(self.disk_file(), data_name,
                                epoch, label_name, value)
        else:
            self.__mm_set_label(data_name, epoch, label_name, value)

    def __mm_set_label(self, data_name, epoch, label_name, value):
        # initialize a new epoch when the label value first being set.
        if epoch not in self.get_epochs(data_name):
            self.__init_epoch(data_name, epoch)
        self.dataset[data_name]['data'][epoch][label_name] = value

    def __df_set_label(self, disk_file, data_name, epoch, label_name, value):
        if epoch not in self.__df_get_epochs(disk_file, data_name):
            self.__df_init_epoch(disk_file, data_name, epoch)
        disk_file['dataset'][data_name]['data'][str(epoch)].\
            attrs[label_name] = value

    def get_label(self, data_name, epoch, label_name):
        '''
        A private matrix for getting label matrix of particular data, epoch 
        and label_name.

        Parameters
        ----------
        data_name : str
            The `data_name` of the data which should be in the 
            `self.dataset.keys()`.
        epoch : int
            The serial number of the epoch in PSG data.
        label_name : str
            The name for distinguishing different label matrices. It is 
            necessary in dataset with multiple labels. The default is 
            \'LABEL\'.

        Returns
        -------
        np.ndarray or array-like
            The label matrix.

        '''
        if self.disk_mode:
            return self.__df_get_label(self.disk_file(), data_name,
                                       epoch, label_name)
        else:
            return self.__mm_get_label(data_name, epoch, label_name)

    def __mm_get_label(self, data_name, epoch, label_name):
        return self.dataset[data_name]['data'][epoch][label_name]

    def __df_get_label(self, disk_file, data_name, epoch, label_name):
        return disk_file['dataset'][data_name]['data'][str(epoch)].\
            attrs[label_name]

    def del_feature(self, feature_name):
        if self.disk_file:
            self.__df_del_feature(self.disk_file(), feature_name)
        else:
            self.__mm_del_feature(feature_name)

    def __mm_del_feature(self, feature_name):
        if len(self.features) > 1:
            for data_name in self.get_data():
                for epoch in self.get_epochs(data_name):
                    self.dataset[data_name]['data'][epoch].pop(feature_name)
            self.features.remove(feature_name)
        else:
            raise Exception('Cannot delete the only feature.')

    def __df_del_feature(self, disk_file, feature_name):
        if len(self.features) > 1:
            for data_name in self.__df_get_data(disk_file):
                for epoch in self.__df_get_epochs(disk_file, data_name):
                    disk_file['dataset'][data_name]['data'][epoch].\
                        pop(feature_name)
            self.features.remove(feature_name)
        else:
            raise Exception('Cannot delete the only feature.')

    def del_label(self, label_name):
        if self.disk_mode:
            self.__df_del_label(self.disk_file(), label_name)
        else:
            self.__mm_del_label(label_name)

    def __mm_del_label(self, label_name):
        for data_name in self.get_data():
            for epoch in self.get_epochs(data_name):
                self.dataset[data_name]['data'][epoch].pop(label_name)
        self.labels.remove(label_name)
        if len(self.labels) == 0:
            self.has_label = False

    def __df_del_label(self, disk_file, label_name):
        for data_name in self.__df_get_data(disk_file):
            for epoch in self.__df_get_epochs(disk_file, data_name):
                disk_file['dataset'][data_name]['data'][epoch].attrs.\
                    pop(label_name)
        self.labels.remove(label_name)
        if len(self.labels) == 0:
            self.has_label = False

    ### STATISTIC ###

    def stat_condition(self, condition_type, data_name = None):
        rst = {}
        if data_name is None:
            data_name = self.get_data()
        for dn in data_name:
            if self.__get_state(dn) != Dataset.ERROR:
                logging.info(dn)
                l = self.get_condition(dn, condition_type)
                if l:
                    if l not in rst.keys():
                        rst[l] = 0
                    rst[l] += 1
        return rst

    def stat_label(self, label_name, label = None ,data_name = None):
        rst = {}
        if data_name is None:
            data_name = self.get_data()
        if label is None:
            label = self.labeldict[label_name].labels()
        elif not isinstance(label, list):
            label = [label]
        for l in label:
            rst[l] = 0
        for dn in data_name:
            if self.__get_state(dn) != Dataset.ERROR:
                logging.info(dn)
                for epoch in self.get_epochs(dn):
                    l = self.get_label(dn, epoch, label_name)
                    if l in label:
                        rst[l] += 1
        return rst

    def stat_plot_bar(stat, show = True, save_path = None):
        # the input is a dict, whose keys are the `label_name` and
        # values are the count
        x = list(range(len(stat)))
        tick_label = [k for k in stat.keys()]
        y = [stat[k] for k in stat.keys()]
        plt.bar(x,y, align = 'center', tick_label = tick_label)
        plt.xlabel('LABEL')
        plt.ylabel('COUNT')
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    def stat_plot_pie(stat, show = True, save_path = None):
        # the input is a dict, whose keys are the `label_name` and
        # values are the count
        kinds = [k for k in stat.keys()]
        values = [stat[k] for k in stat.keys()]
        plt.axes(aspect='equal') # ensure the pie is a circle
        plt.pie(values, autopct = '%3.1f%%', labels = kinds)
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    ### SOURCE ###

    def __df_init_source(self):
        self.sources = {}

    def __df_init_source_data(self, data_name):
        if not hasattr(self, 'sources'):
            self.__df_init_source()
        self.sources[data_name] = {}

    def __df_has_source_data(self, data_name):
        if not hasattr(self, 'sources'):
            return False
        else:
            return data_name in self.sources

    def __set_source(self, data_name, source_name, value):
        if self.disk_mode:
            self.__df_set_source(data_name, source_name, value)
        else:
            self.__mm_set_source(data_name, source_name, value)

    def __mm_set_source(self, data_name, source_name, value):
        self.dataset[data_name]['source'][source_name] = value

    def __df_set_source(self, data_name, source_name, value):
        if not self.__df_has_source_data(data_name):
            self.__df_init_source_data(data_name)
        self.sources[data_name][source_name] = value

    def __has_source(self, data_name, source_name):
        if self.disk_mode:
            return self.__df_has_source(data_name, source_name)
        else:
            return self.__mm_has_source(data_name, source_name)

    def __mm_has_source(self, data_name, source_name):
        rst = source_name in self.dataset[data_name]['source']
        return rst

    def __df_has_source(self, data_name, source_name):
        rst = source_name in self.sources[data_name]
        return rst

    def get_source(self, data_name, source_name = None):
        if self.disk_mode:
            return self.__df_get_source(data_name, source_name)
        else:
            return self.__mm_get_source(data_name, source_name)

    def __mm_get_source(self, data_name, source_name = None):
        if source_name is None:
            return self.dataset[data_name]['source']
        else:
            return self.dataset[data_name]['source'][source_name]

    def __df_get_source(self, data_name, source_name = None):
        if source_name is None:
            return self.sources[data_name]
        else:
            return self.sources[data_name][source_name]
    
    def get_relative_path(self, absolute_path):
        relative_path = os.path.relpath(absolute_path, self.path)
        return relative_path

    def get_absolute_path(self, relative_path):
        absolute_path = os.path.join(self.path, relative_path)
        absolute_path = os.path.normpath(absolute_path)
        return absolute_path

    ### STATE ###

    def __set_state(self, data_name, state = None):
        '''
        A private method for setting the state of data.

        Parameters
        ----------
        data_name : str
            The `data_name` of the data which should be in the 
            `self.dataset.keys()`.
        state : int, optional
            The state to be set. Typically `Dataset.ADDED`, `Dataset.CHECKED`, 
            `Dataset.PREPROCESSED`, `Dataset.PREDICTED` and `Dataset.ERROR`.
            The default is None which will change the state to the next state.
            e.g., changing `Dataset.ADDED` to `Dataset.CHECKED`.

        Returns
        -------
        None.

        '''
        if self.disk_mode:
            self.__df_set_state(self.disk_file(), data_name, state)
        else:
            self.__mm_set_state(data_name, state)
        
    def __mm_set_state(self, data_name, state = None):
        if state is None:
            self.dataset[data_name]['STATE'] += 1
        else:
            self.dataset[data_name]['STATE'] = state

    def __df_set_state(self, disk_file, data_name, state = None):
        if state is None:
            disk_file['dataset'][data_name].attrs['STATE'] += 1
        else:
            disk_file['dataset'][data_name].attrs['STATE'] = state

    def __get_state(self, data_name, return_str = False):
        '''
        A private method for getting the state of data.

        Parameters
        ----------
        data_name : str
            The `data_name` of the data which should be in the 
            `self.dataset.keys()`.
        return_str : bool, optional
            If True, the state will return in type `str`. The default is False 
            which returns `int`.

        Returns
        -------
        int or str
            The state of the data. The type of returns depends on the 
            `return_str` parameter.

        '''
        if self.disk_mode:
            return self.__df_get_state(self.disk_file(), data_name, return_str)
        else:
            return self.__mm_get_state(data_name, return_str)

    def __mm_get_state(self, data_name, return_str = False):
        state = {0 : 'ADDED',
                 1 : 'CHECKED',
                 2 : 'PREPROCESSED',
                 3 : 'PREDICTED',
                 -1: 'ERROR'
                 }
        if return_str:
            return state[self.dataset[data_name].get('STATE')]
        else:
            return self.dataset[data_name].get('STATE')

    def __df_get_state(self, disk_file, data_name, return_str = False):
        state = {0 : 'ADDED',
                 1 : 'CHECKED',
                 2 : 'PREPROCESSED',
                 3 : 'PREDICTED',
                 -1: 'ERROR'
                 }
        if return_str:
            return state[disk_file['dataset'][data_name].attrs['STATE']]
        else:
            return disk_file['dataset'][data_name].attrs['STATE']

    ### PREPROCESS ###

    def prepare_feature(self, data_name, procedure, feature_name = 'FEATURE', 
                        max_epoch = math.inf):
        # 2020-3-1 预处理数据
        method = procedure.procedure()
        i = 0
        while i < math.inf:
            try:
                epoch, value = self.get_source(data_name, feature_name).get()
                value = method(value)
                self.__set_feature(data_name, epoch, feature_name, value)
            except StopIteration:
                break
            except MemoryError:
                # handle the OOM problem
                print('======================================================'
                      '======')
                print('Now we encounter an OUT OF MEMORY error.')
                print('The memory usage of this dataset is approximately'
                      ' {0:.3f} GB'.format(self.memory_usage()))
                print('Now we are at Iteration {0} of \'{1}\'.'\
                      .format(i, data_name))
                print('You can read the log file for more details.')
                print('The possible solution:')
                print('\t- Use a more powerful computer which has larger '
                      'memory.')
                print('\t- Lower your data\'s sampling rate.')
                print('\t- Use less channels.')
                print('\t- Add less data sources.')
                print('======================================================='
                      '=====')
                print('INTERRUPTED')
            except:
                tb = traceback.format_exc()
                self.exception_logger.submit(data_name, i, feature_name, tb)
            finally:
                i += 1
        self.get_source(data_name, feature_name).clean()

    def prepare_label(self, data_name, label_name = 'LABEL', 
                      max_epoch = math.inf):
        # 2020-3-1 预处理标签
        i = 0
        while i < math.inf:
            try:
                epoch, value = self.get_source(data_name, label_name).get()
                self.__set_label(data_name, epoch, label_name, value)
            except StopIteration:
                break
            except:
                tb = traceback.format_exc()
                self.exception_logger.submit(data_name, i, label_name, tb)
            finally:
                i += 1
        self.get_source(data_name, label_name).clean()

    def source_check(self, data_name):
        rst = True
        if self.__get_state(data_name) == Dataset.ERROR:
            rst = False
            return rst
        for f in self.features:
            if not self.__has_source(data_name, f):
                tb = 'Feature ' + f + 'not exist in data ' + data_name + '.'
                self.exception_logger.submit(data_name, None, f, tb)
                rst = False
        if self.has_label:
            for l in self.labels:
                if not self.__has_source(data_name, l):
                    tb = 'Label ' + l + 'not exist in data ' + data_name + '.'
                    self.exception_logger.submit(data_name, None, l, tb)
                    rst = False
        if rst:
            self.__set_state(data_name, Dataset.CHECKED)
        else:
            self.__set_state(data_name, Dataset.ERROR)
        return rst

    def epoch_check(self, data_name):
        rst = True
        to_be_del = []
        if self.__get_state(data_name) == Dataset.ERROR:
            rst = False
            return rst
        elif len(self.get_epochs(data_name)) == 0:
            self.__set_state(data_name, Dataset.ERROR)
            rst = False
            return rst
        for epoch in self.get_epochs(data_name):
            p = False  # pop this epoch or not
            for f in self.features:
                if f not in self.__get_epoch_element_list(data_name, epoch):
                    self.exception_logger.submit(data_name, epoch, f, 
                                                 'Feature is not in dataset.')
                    p = True
            if self.has_label:
                for l in self.labels:
                    if l not in self.__get_epoch_element_list(data_name,
                                                              epoch):
                        self.exception_logger.\
                            submit(data_name, epoch, l, 
                                   'Label is not in dataset.')
                        p = True
            if p:
                to_be_del.append(epoch)
        for e in to_be_del:
            self.__del_epoch(data_name, e)
        if rst:
            self.__set_state(data_name, Dataset.PREPROCESSED)
        else:
            self.__set_state(data_name, Dataset.ERROR)
        return rst

    def shape_check(self):
        # check if every data's state is ERROR
        all_error = True
        for data_name in self.get_data():
            if self.__get_state(data_name) != Dataset.ERROR:
                all_error = False
        if all_error:
            raise Exception('Preprocessing error.')
        # check if features' shape is consistent.
        self.shape = {}
        for data_name in self.get_data():
            # 检查有没有错误，如果没有错误，再进一步检查
            if self.__get_state(data_name) != Dataset.ERROR:
                # 检查shape一致性
                e = False  # data is error or not
                for epoch in self.get_epochs(data_name):
                    # features
                    for f in self.features:
                        if f not in self.shape.keys():
                            self.shape[f] = self.get_feature(data_name, 
                                                          epoch, f).shape
                        else:
                            tb = 'Shape is inconsistent.'
                            if self.shape[f] != self.get_feature(data_name, 
                                                          epoch, f).shape:
                                self.exception_logger.submit(data_name, epoch,
                                                             f, tb)
                                e = True
                if e:
                    self.__set_state(data_name, Dataset.ERROR)

    def preprocess(self, procedure):
        for data_name in self.get_data():
            if self.source_check(data_name):
                if not isinstance(procedure, list):
                    for f in self.features:
                        self.prepare_feature(data_name, procedure, f)
                else:
                    for f, p in procedure:
                        self.prepare_feature(data_name, p, f)
                if self.has_label:
                    for l in self.labels:
                        self.prepare_label(data_name, l)
                self.epoch_check(data_name)
        self.shape_check()

    ### SAMPLE ###

    def stat_classes(self, elements, unit=None):
        '''
        Add up different classes (or class group) according to input elements.
        Return a dict whose `keys()` are classes and `values()` are epochs for
        each class.

        >>> # compute the number of classes
        >>> len(ds.stat_classes(elements).keys())

        >>> # list all the classes
        >>> list(ds.stat_classes(elements).keys())

        Parameters
        ----------
        elements : str, list or dict
            `element_name` or a list of them. Also can be a dict with format:
            >>> {element_name : [element_value, element_value]}
            to add up specific classes.

            For example, you want to add up classes whose `'LABEL'` is '1' or
            '2', and `'DIAGNOSE'` is `'healthy'`. You can use:
            >>> dataset.stat_classes(
            ...    {'LABEL' : ['1', '2'],
            ...     'DIAGNOSE' : 'healthy'})
            
            If you want to add up all the `'LABEL'` but just when `'DIAGNOSE'`
            is `'healthy'`. You can use:
            >>> dataset.stat_classes(
            ...    {'LABEL' : [],  # an empty list means use all the values
            ...     'DIAGNOSE' : 'healthy'})
            Notice that an empty list means use all the values.

            Elements must be `label_name` or `condition_type`.

        Raises
        ------
        TypeError
            Raised when input unsupported parameters.
        ValueError
            Raised when the input element do not exist in the dataset.
            Or Raised when the input element is not `label_name` or
            `condition_type`

        Returns
        -------
        rst : dict
            The key is `str` or `tuple`, represents a class or class group.
            The value is a list of `data_name` or a list of 
            `(data_name, epoch)` tuples.

        '''
        # check input
        select = False  # to control whether to use selected elements or all
        unit_candidate = ['epoch', 'data']
        if unit is not None and unit not in unit_candidate:
            raise ValueError('Invalid unit. Only \'epoch\' or \'data\' allowd')
        if isinstance(elements, str):
            elements = [elements]
        elif isinstance(elements, list):
            pass
        elif isinstance(elements, dict):
            select = True
            for key in elements:
                if isinstance(elements[key], str):
                    elements[key] = [elements[key]]
        else:
            raise TypeError('The input parameter `elements` must be `str`, '
                            '`list` or `dict`. It represents label names or '
                            'condition types.')

        num = len(elements)  # numbers of elements

        # assert elements must be `label` or `condition`
        # and separate labels and conditions
        condition = []
        label = []
        for element in elements:
            if element not in self.elements:
                raise ValueError(element + ' not in this dataset.')
            elif self.elements[element] == 'label':
                label.append(element)
            elif self.elements[element] == 'condition':
                condition.append(element)
            else:
                raise ValueError('The input parameter `elements` must be label'
                                 ' names or condition types.')
        # check passed
        rst = dict()
        if unit is None:
            if len(label) == 0:
                unit = 'data'
            else:
                unit = 'epoch'

        if unit == 'data':
            for d in self.get_data():  # traversal and add up classes
                fit = True
                c = []
                for cond in condition:
                    if select and elements[cond] and self.get_condition(d, cond) \
                        not in elements[cond]:  # encounter an invalid condition
                        fit = False
                        break  # drop this data
                    c.append(self.get_condition(d, cond))
                if not fit:
                    continue
                c = tuple(c) if num > 1 else c[0]  # use tuple or single
                                                   # element
                if c not in rst:
                    rst[c] = []
                rst[c].append(d)
        else:  # the unit is `epoch`
            for d in self.get_data():
                for e in self.get_epochs(d):
                    fit = True
                    c = []
                    for cond in condition:
                        if select and elements[cond] and self.get_condition(d, \
                            cond) not in elements[cond]:  # encounter an invalid condition
                            fit = False
                            break  # drop this epoch
                        c.append(self.get_condition(d, cond))
                    for l in label:
                        if select and elements[l] and self.get_label(d, e, l) \
                            not in elements[l]:  # encounter an invalid label
                            fit = False
                            break  # drop this data
                        c.append(self.get_label(d, e, l))
                    if not fit:
                        continue
                    c = tuple(c) if num > 1 else c[0]  # use tuple or single 
                                                       # element
                    if c not in rst:
                        rst[c] = []
                    rst[c].append((d, e))
        return rst

    def sample_data(self, data_name, lst, tmin=0, tmax=0, data_padding=True,
                    max_len=None, epoch_padding=False, test_data_name=None,
                    autoencoder=False, array_type=int, concat:bool=False,
                    x_dict: Dict[str, BaseDict] = {},
                    y_dict: Dict[str, BaseDict] = {}):
        if test_data_name is None:
            test_data_name = data_name

        x, y = lst
        x_samp = []
        if isinstance(x, str):
            x = [x]
        for i in x:
            x_samp.append(self.sample_serial_x(data_name, i, tmin, tmax,
                                               data_padding, max_len,
                                               epoch_padding,
                                               array_type=array_type,
                                               concat=concat, x_dict=x_dict))
        if len(x_samp) == 1:
            x_samp = x_samp[0]
        else:
            x_samp = np.asarray(x_samp)
        # y
        if autoencoder:
            return (x_samp, x_samp)
        elif y is None:
            return x_samp
        else:
            y_samp = []
            if isinstance(y, str):
                y = [y]
            for i in y:
                y_samp.append(
                    # TODO: v0.2.62
                    self.__samp_condtion(test_data_name, i,
                                         array_type, y_dict))
                    # self.labeldict[i].trans(
                    #     self.get_condition(test_data_name, i),
                    #     array_type=array_type))
            if len(y_samp) == 1:
                y_samp = y_samp[0]
            else:
                y_samp = np.asarray(y_samp)
            return (x_samp, y_samp)

    def sample_epoch(self, data_name, epoch, lst, tmin=0, tmax=0,
                    epoch_padding=False, test_data_name=None, test_epoch=None,
                    autoencoder=False, array_type=int, concat:bool=False,
                    x_dict: Dict[str, BaseDict] = {},
                    y_dict: Dict[str, BaseDict] = {}):
        if test_data_name is None:
            test_data_name = data_name
            test_epoch = epoch

        x, y = lst
        x_samp = []
        if isinstance(x, str):
            x = [x]
        for i in x:
            x_samp.append(self.sample_epoched_x(data_name, epoch, i,
                                                tmin, tmax, epoch_padding,
                                                array_type=array_type,
                                                concat=concat, x_dict=x_dict))
        if len(x_samp) == 1:
            x_samp = x_samp[0]
        else:
            x_samp = np.asarray(x_samp)
        # y
        if autoencoder:
            return (x_samp, x_samp)
        elif y is None:
            return x_samp
        else:
            y_samp = []
            if isinstance(y, str):
                y = [y]
            for i in y:
                y_samp.append(self.sample_epoched_y(test_data_name, test_epoch,
                                                    i, array_type=array_type,
                                                    y_dict=y_dict))
            if len(y_samp) == 1:
                y_samp = y_samp[0]
            else:
                y_samp = np.asarray(y_samp)
            return (x_samp, y_samp)

    def sample_epoched_x(self, data_name:str, epoch:int, element_name:str,
                         tmin:int=0, tmax:int=0, padding:bool=False,
                         array_type:type=int, concat:bool=False,
                         x_dict: Dict[str, BaseDict] = {}):
        '''
        A low-level method to sample one epoch according to parameters.

        Parameters
        ----------
        data_name : str
            The target datum.
        epoch : int
            The target epoch.
        element_name : str
            The target element, it may be a feature_name, a label_name or a
            condition_name.
        tmin : int, optional
            The number of preceding epochs before the target epoch to be
            sampled. Usually used for RNN-related model or other models related
            to time series. The default is 0, means do not sample any other
            epochs.
        tmax : int, optional
            The number of following epochs after the target epoch to be
            sampled. Usually used for RNN-related model or other models related
            to time series. The default is 0, means do not sample any other
            epochs.
        padding : bool, optional
            At the beginning or end of the datum, whether the target epoch to
            be padded with zero or dropped. If True, they will be padded, 
            otherwise dropped. The default is False.
        array_type : type or None, optional
            The type of elements in return array. The default and most
            recommended is int. You should input directly a `type`, not a
            `str`.
            For example:
                array_type=bool          (Correct)
                array_type='bool'        (Wrong)

        concat : bool, optional
            Decide the output shape when tmin!=0 or tmax!=0, i.e., multiple
            time steps are sampled.
            If you have not set the `tmin` or `tmax`. You do not need to care
            about this parameter.
            The default is False, means the output will not concatenate time
            steps (i.e. sampled epochs) but leave the output shape as
            `(time_step, channel[, feature-related])`. In this situation, every
            time step occupies a unique matrix. It is suitable for RNN-related
            models.
            If `concat==True`, all the sampled epoches will concatenate by the
            'time' axis. It results in continuous feature which is suitable for
            TCN or other CNN-related models.

        Raises
        ------
        DataStateError
            Raised when your data have not preprocessed yet. Check your data
            state.
        BrokenTimestepError
            The flag of the end of one datum. It is for the iterators. You need
            not do any treatment about it.
        ValueError
            The shape mismatch.

        Returns
        -------
        np.ndarray
            The sampled x.

        '''
        TIMEAXIS = 1  # set the time axis
        # Skip this epoch if returns `None`
        logging.info('== EPOCHED SAMPLE ==')
        # check state
        if self.__get_state(data_name) < Dataset.PREPROCESSED:
            raise DataStateError('You cannot sample from a data not correctly '
                                 'preprocessed. The target data `{0}` has a '
                                 'state `{1}`.'.format(data_name,
                                                       self.__get_state(
                                                           data_name, True)))
        logging.debug('Data_name: {0}'.format(data_name))
        logging.debug('Epoch: {0}'.format(str(epoch)))
        # ===== without timestep =====
        if tmin == 0 and tmax == 0:
            # == feature ==
            if self.elements[element_name] == 'feature':
                return self.get_feature(data_name, epoch, element_name)
            # == label ==
            elif self.elements[element_name] == 'label':
                # TODO: v0.2.62
                return self.__samp_label(data_name, epoch,
                                         element_name, array_type, x_dict)
                # return self.labeldict[element_name].trans( \
                #         self.get_label(data_name, epoch, element_name),
                #         array_type=array_type)
        # ===== with timestep =====
        else:
            # check timespan
            # timespan cannot be longer than data_length
            data_length = len(self.get_epochs(data_name))
            logging.debug('Data Length: ' + str(data_length))
            assert tmax >= 0
            timespan = abs(tmin) + tmax + 1
            assert timespan <= data_length
            # sample
            epoch_list = self.get_epochs(data_name)
            logging.debug('The epoch between [{0}, {1}]'.format(epoch_list[0],
                                                            epoch_list[-1]))
            idx = epoch_list.index(epoch)
            # if `idx - abs(tmin) < 0`, it will cause problem in slicing
            # so check if `idx - abs(tmin) < 0`
            lower = idx - abs(tmin) if idx - abs(tmin) >= 0 else 0
            upper = idx + tmax + 1
            # == feature ==
            if self.elements[element_name] == 'feature':
                dtype = 'float32'
                sample_shape = self.shape[element_name]
                r = np.asarray([self.get_feature(data_name, i, element_name) 
                              for i in epoch_list[lower : upper]])
            # == label ==
            # TODO: v0.2.62
            elif self.elements[element_name] == 'label':
                dtype = 'int32'
                sample_shape = self.labeldict[element_name].shape()
                r = np.asarray([self.__samp_label(data_name, i,
                                                  element_name, array_type,
                                                  x_dict)
                        for i in epoch_list[lower : upper]])
                # r = np.asarray([self.labeldict[element_name].trans( \
                #         self.get_label(data_name, i, element_name),
                #         array_type=array_type)
                #         for i in epoch_list[lower : upper]])
            if len(r) < timespan:
                if not padding:
                    raise BrokenTimestepError()
                else:
                    x = np.full((timespan, ) + sample_shape, 0, dtype=dtype)
                    trunc = r[-timespan:]
                    # check `trunc` has expected shape
                    trunc = np.asarray(trunc, dtype=dtype)
                    if trunc.shape[1:] != sample_shape:
                        raise ValueError('Shape of sample {0} of sequence is '
                                         'different from expected shape {1}'.\
                                        format(trunc.shape[1:], sample_shape))
                    if idx - abs(tmin) >= 0:
                        x[:len(trunc)] = trunc
                    elif idx - abs(tmin) < 0:
                        x[-len(trunc):] = trunc
                    r = x
            # == concat ==
            # Only when `x` is a feature, the concat is needed.
            if concat and self.elements[element_name] == 'feature':
                r = np.concatenate(r, axis=TIMEAXIS)
            logging.debug(r.shape)
            return r

    def sample_epoched_y(self, data_name, epoch, element_name, array_type=int,
                         y_dict: Dict[str, BaseDict] = {}):
        # check state
        if self.__get_state(data_name) < Dataset.PREPROCESSED:
            raise DataStateError('You cannot sample from a data not correctly '
                                 'preprocessed. The target data `' + data_name
                                 + '` has a state `' 
                                 + self.__get_state(data_name, True) + '`.')
        # label
        # TODO: v0.2.62
        if self.elements[element_name] == 'label':
            return self.__samp_label(data_name, epoch, element_name,
                                     array_type, y_dict)
            # return self.labeldict[element_name].trans( \
            #             self.get_label(data_name, epoch, element_name),
            #             array_type=array_type)
        # TODO: v0.2.62
        elif self.elements[element_name] == 'condition':
            return self.__samp_condition(data_name, element_name,
                                         array_type, y_dict)
            # return self.labeldict[element_name].trans( \
            #             self.get_condition(data_name, 
            #                                condition_type=element_name),
            #             array_type=array_type)
        else:
            raise ValueError('The input element ' + element_name + 'is a `'
                             + self.elements[element_name] + '`. It cannot'
                             'act as `epoched_y` sample.')
    
    def sample_serial_x(self, data_name, element_name, tmin, tmax,
                        data_padding, max_len, epoch_padding, array_type=int,
                        concat:bool=False, x_dict: Dict[str, BaseDict] = {}):
        timespan = abs(tmin) + tmax + 1
        rst = []
        for epoch in self.get_epochs(data_name):
            try:
                rst.append(self.sample_epoched_x(data_name, epoch, 
                                                 element_name,
                                                 tmin, tmax, epoch_padding,
                                                 array_type=array_type,
                                                 concat=concat,
                                                 x_dict=x_dict))
            except BrokenTimestepError:
                continue
        if data_padding:
            if max_len is None:
                raise ValueError('A certain `max_len` is needed for '
                                 'data_padding')
            if self.elements[element_name] == 'feature':
                if timespan == 1:
                    sample_shape = self.shape[element_name]
                else:
                    sample_shape = (timespan, ) + self.shape[element_name]
                dtype = 'float32'
            else:
                if timespan == 1:
                    sample_shape = self.labeldict[element_name].shape()
                else:
                    sample_shape = (timespan, ) + \
                        self.labeldict[element_name].shape()
                dtype = 'int32'
            x = np.full((max_len, ) + sample_shape, -1, dtype=dtype)
            trunc = rst[-max_len:]
            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence '
                                 'is different from expected shape %s' %
                                 (trunc.shape[1:], sample_shape))
            x[-len(trunc):] = trunc
            rst = x
        else:
            rst = np.asarray(rst)
        return rst

    def __samp_condtion(self, data_name: str, condition_type: str,
                        array_type: type,
                        labeldict: Dict[str, BaseDict] = {}) -> np.ndarray:
        if condition_type in labeldict:
            return labeldict[condition_type].trans(
                self.get_condition(data_name, condition_type),
                array_type=array_type)
        else:
            return self.labeldict[condition_type].trans(
                self.get_condition(data_name, condition_type),
                array_type=array_type) 

    def __samp_label(self, data_name: str, epoch: int, label_type: str,
                     array_type: type,
                     labeldict: Dict[str, BaseDict] = {}) -> Union[int, np.ndarray]:
        if label_type in labeldict:
            return labeldict[label_type].trans(
                self.get_label(data_name, epoch, label_type),
                array_type=array_type)
        else:
            return self.labeldict[label_type].trans(
                self.get_label(data_name, epoch, label_type),
                array_type=array_type)

    def memory_usage(self, unit = 'GB'):
        if self.disk_mode:
            return 0
        else:
            r = total_size(self.dataset)
            if unit.upper() == 'KB':
                r /= 1024
            elif unit.upper() == 'MB':
                r /= 1024 * 1024
            else:
                r /= 1024 * 1024 * 1024
            return r

    ### LOAD & SAVE ###

    def __df_save_attr_0_2(self, disk_file):
        attrs = ['VERSION', 'name', 'comment']
        for attr in attrs:
            if hasattr(self, attr):
                disk_file.attrs[attr] = getattr(self, attr)

    def __df_save_label_dict_0_2(self, disk_file):
        # 保存标签字典
        if self.has_label:
            disk_file.create_group('label_dict')
            for label_name in self.labeldict.keys():
                disk_file['label_dict'].create_group(label_name)
                tem = self.labeldict[label_name].dict
                for label in tem:
                    disk_file['label_dict'][label_name].create_dataset(label, data = tem[label]) # 2020-3-5 存的是int

    def __df_save_dataset_0_2(self, disk_file):
        self.__df_init_dataset(disk_file)
        for data_name in self.get_data():
            if self.__get_state(data_name) == Dataset.PREPROCESSED:
                self.__df_init_new_data(disk_file, data_name)
                self.__df_save_condition_0_2(disk_file, data_name)
                # 保存数据 2020-3-5
                self.__df_save_data_0_2(disk_file, data_name)

    def __df_save_condition_0_2(self, disk_file, data_name):
        l = self.get_condition(data_name)
        if l:
            for condition_type in l.keys():
                self.__df_set_condition(disk_file, data_name, condition_type, l[condition_type])
 
    def __df_save_source_0_2(self, disk_file, data_name):
        # 暂时先不用存source
        disk_file['dataset'][data_name].create_group('source')
        source = self.get_source(data_name)
        for source_name in source.keys():
            disk_file['dataset'][data_name]['source'].create_group(source_name)
            disk_file['dataset'][data_name]['source'].attrs['abs'] = self.get_source(data_name, source_name).path
            disk_file['dataset'][data_name]['source'].attrs['rel'] = self.get_source(data_name, source_name).path
            disk_file['dataset'][data_name]['source'].attrs['source_type'] = self.get_source(data_name, source_name).source_type

    def __df_save_data_0_2(self, disk_file, data_name):
        for epoch in self.get_epochs(data_name):
            self.__df_init_epoch(disk_file, data_name, epoch)
            for feature in self.features:
                self.__df_set_feature(disk_file, data_name, epoch, feature, 
                                      self.get_feature(data_name, epoch, feature))
            # 保存标签 2020-3-5
            if self.has_label:
                for label in self.labels:
                    self.__df_set_label(disk_file, data_name, epoch, label, 
                               self.get_label(data_name, epoch, label))

    def df_load_label_dict_0_2(self, disk_file):
        # 获取标签字典
        if 'label_dict' in disk_file.keys():
            for label_name in disk_file['label_dict'].keys():
                self.labeldict[label_name] = ClassDict()
                for label in disk_file['label_dict'][label_name].keys():
                    self.labeldict[label_name].add(label, int(disk_file['label_dict'][label_name][label][()])) # 2020-3-5 存的是int

    def df_load_dataset_0_2(self, disk_file):
        # 获取数据
        for data_name in disk_file['dataset'].keys():
            self.__init_new_data(data_name)
            self.__df_load_condition_0_2(disk_file, data_name)
            self.__set_state(data_name, Dataset.PREPROCESSED)
            for epoch in self.__df_get_epochs(disk_file, data_name):
                self.__init_epoch(data_name, int(epoch))
                for feature_name in disk_file['dataset'][data_name]['data'][str(epoch)].keys():
                    self.__new_feature(feature_name)
                    self.__set_feature(data_name, int(epoch), feature_name,
                                       self.__df_get_feature(disk_file, data_name, 
                                                      epoch, feature_name))
                for label_name in disk_file['dataset'][data_name]['data'][str(epoch)].attrs:
                    self.has_label = True
                    self.__new_label(label_name)
                    self.__set_label(data_name, int(epoch), label_name,
                                     self.__df_get_label(disk_file, data_name, epoch, 
                                                  label_name))

    def __df_load_condition_0_2(self, disk_file, data_name):
        if self.__df_has_condition(disk_file, data_name):
            l = self.__df_get_condition(disk_file, data_name)
            for condition_type in l.keys():
                self.__new_condition(condition_type)
                self.set_condition(data_name, condition_type, l[condition_type])

    def df_load_source_0_2(self, disk_file, data_name):
        pass

    @staticmethod
    def load(path, mode = 'memory', load_source = False):
        '''
        Load the existed dataset from disk.

        Parameters
        ----------
        path : path-like
            The path to the dataset file.
        mode : {'memory', 'disk'}, optional
            For the faster speed, 'memory' mode will process all the data in
            the memory. But if you have a huge dataset which exceeds or will
            exhaust your memory, the 'disk' mode will cache all the data at
            disk until you are to use them. It do slow down the computation,
            but it is a good resolution when you are not able to extend your
            memory. The default is 'memory'.
        load_source : bool, optional
            Whether to load the sources of dataset. The default is False.

        Returns
        -------
        dataset : sleepology.dataset.Dataset

        '''
        # 2020-3-1 hdf5->dict
        disk_file = h5py.File(path, 'r')
        # 获取属性
        VERSION = disk_file.attrs['VERSION']
        name = disk_file.attrs['name']
        comment = disk_file.attrs['comment']
        dir_name = os.path.dirname(path)
        dataset = Dataset(name, dir_name, comment = comment, mode = mode)
        dataset.VERSION = VERSION
        
        dataset.df_load_label_dict_0_2(disk_file)
        dataset.df_load_dataset_0_2(disk_file)
        
        disk_file.close()
        return dataset

    def save(self, path = None):
        # 2020-2-23 dict->hdf5
        save_path = path if path is not None else self.save_direction
        print('Saved at:')
        print(save_path)
        disk_file = h5py.File(save_path, 'w')
        self.__df_save_attr_0_2(disk_file)
        self.__df_save_label_dict_0_2(disk_file)
        self.__df_save_dataset_0_2(disk_file)
        disk_file.close()