# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:31:23 2020

@author: chizh
"""
from .source import *
from .utils import total_size
from .exceptions import DataStateError, BrokenTimestepError

import h5py
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
import math
import os
import json
import logging
import traceback
import datetime

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

package_root = os.path.dirname(os.path.abspath(__file__))

                               
class ExceptionLogger(object):
    def __init__(self, dataset):
        self.create = False
        self.dataset = dataset

    def first_line(self):
        time = datetime.datetime.now()
        postfix = time.strftime('%Y%m%d%H%M%S')
        fname = 'log_' + postfix + '.log'
        self.dst = os.path.join(self.dataset.path, fname)
        print('The log will be saved at')
        print(self.dst)
        wrapper = '\n'
        line = '============================================================='
        self.create = True
        with open(self.dst, 'w') as f:
            f.write('ERROR LOG' + wrapper)
            f.write(line + wrapper)
            f.write(wrapper)
            f.write('Dataset Name: ' + self.dataset.name + wrapper)
            f.write('Time: ' + str(time) + wrapper)
            f.write(line + wrapper)

    def submit(self, data_name, epoch, feature_or_label_name, trace_back):
        if not self.create:
            self.first_line()
        wrapper = '\n'
        tab = '\t'
        line = '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
        with open(self.dst, 'a') as f:
            f.write(wrapper)
            f.write(wrapper)
            f.write('DATA: ' + data_name + wrapper)
            if epoch is not None:
                f.write('EPOCH: ' + str(epoch) + tab)
            if feature_or_label_name is not None:
                f.write('SERIES: ' + feature_or_label_name + wrapper)
            f.write(trace_back + wrapper)
            f.write(line + wrapper)


class Procedure(object):
    '''
        用于生成并保存对于每一个通道的处理方式、以及保存哪些通道。（只保留通道而不经任何处理的，
        preprocessing_method为accept） 2020-2-18
        self.__proce = ['preprocessing_method', [channel_list], [param_list]]
        self.__proce[i]是每一个条目
        self.__proce[i][0]是方法名
        self.__proce[i][1]是通道列表
        self.__proce[i][2]是参数列表
    '''
    def __init__(self, args = None):
        '''
        2020-3-3
        args 可能有四种形式：
        ①list的话，可能是直接输入self.__proce列表
        ②function的话，可能是直接输入self.procedure函数(一定要输入高阶函数！)
        ③str的话，可能是存储Procedure的文件
        ④None的话，就是空白Procedure
        '''
        self.__proce = list()
        self.channel_list = set() # 2020-2-19 整理一个不重复的通道列表
        if args is not None:
            if isinstance(args, str):
                self.load(os.path.abspath(args))
            elif isinstance(args, list):
                self.set_procedure(args)
            elif callable(args):
                self.procedure = args
            else:
                raise TypeError('Procedure class connot be initialized by given args')
    
    def add_method(self, method, chan_list, param_list = None):
        # 2020-2-13 为已经存在的通道添加操作列表
        if not isinstance(method, str) or not isinstance(chan_list, list) \
            or not isinstance(param_list, list) and param_list is not None:
            raise TypeError()
        else:
            self.__proce.append([method, chan_list, param_list])
            for channel in chan_list:
                self.channel_list.add(channel)
    
    def clear(self):
        # 2020-2-19 重新初始化
        self.__proce = list()
        self.channel_list = set()
    
    def add_channels(self, channels):
        if not isinstance(channels, list):
            channels = [channels]
        for channel in channels:
            self.channel_list.add(channel)
    
    def get_channels(self):
        # 2020-2-13 获取已经存在的通道列表
        return list(self.channel_list)
    
    def set_procedure(self, proce): # 2020-2-19 直接接收格式化后的list，而不一个一个method添加
        self.__proce = proce
        # 添加通道列表 2020-2-19
        for method in self.__proce:
            for channel in method[1]:
                self.channel_list.add(channel)
    
    def get_procedure(self):
        return self.__proce
    
    def procedure(self):
        # 2020-3-1 高阶函数，返回一个处理流程。低阶函数的输入是raw，输出是np.ndarray
        ####################  未完待续  ####################
        def pipeline(data):
            for preprocessing_method, channel_list, param_list in self.__proce:
                pass
                #data = getattr(preprocess, preprocessing_method)(data, 
                #              param_list, picks = channel_list)
            return data
        return pipeline  
    
    def dumps(self):
        return json.dumps(self.__proce)
    
    def load(self, load_path):
        # 2020-2-13 从指定路径加载步骤
        with open(load_path, 'r') as f:
            self.__proce = json.load(fp = f)
            # 添加通道列表 2020-2-19
            for method in self.__proce:
                for channel in method[1]:
                    self.channel_list.add(channel)
    
    def save(self, save_path):
        # 2020-2-13 将步骤保存至指定路径
        with open(save_path, 'w') as f:
            json.dump(self.__proce, f)

class LabelDict(object):
    #  2020-3-2 标签字典
    def __init__(self, load_path = None):
        # 2020-3-2 array存的是数字
        self.label2array = {}
        self.array2label = {}
        self.length = 0 # 2020-3-2 提供当前字典的长度信息，也是one_hot数组的长度参考
        if load_path is not None:
            self.load(load_path)

    def shape(self):
        return (self.length, )

    def labels(self):
        return list(self.label2array.keys())

    def add_label(self, label, order = None):
        # 2020-3-2
        array = self.length
        if order is not None and isinstance(order, int):
            array = order
        self.label2array[label] = array
        self.array2label[array] = label
        self.length += 1
    
    def get_array(self, label, array_type = int):
        '''
        Get one-hot or N-hot array according to given label. If `array_type` is
        `None`, the function will return a number directly.

        **ONLY FOR INTERFACE, NOT RECOMMENDED**
        If label is `None`, it represents a `padding` is requested. An array
        filled with zero (if `array_type` is not None) or `-1` (if `array_type`
        is None) will be return.

        Parameters
        ----------
        label : str or list or None
            A label in this LabelDict or a list of labels. If None, return a
            padding array or `-1`.
        array_type : type or None, optional
            The type of elements in return array. The default and most
            recommended is int. You should input directly a `type`, not a
            `str`.
            For example:
                array_type=bool          (Correct)
                array_type=\'bool\'      (Wrong)

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
        # if array_type is None, return a number directly
        if array_type is None:
            if isinstance(label, Iterable):
                if len(label) != 1:
                    raise TypeError(
                        'make sure your request is single-label when'
                        ' array_type is None.')
            elif label is None:  # request padding
                return -1  # `-1` represents padding
            else:
                return self.label2array[label]
        # return N-hot array
        else:
            if isinstance(label, Iterable) and len(label) == 0:
                raise TypeError('at least 1 label should be given.')
            elif label is None:  # request padding
                arr = np.array([False] * self.length)
                return arr.astype(array_type)
            else:
                arr = np.array([False] * self.length)
                for l in label:
                    arr[self.label2array[l]] = True
                return arr.astype(array_type)

    def get_label(self, array):
        if len(array) != self.length:
            raise TypeError('array length dismatch')
        arr = array.astype(bool)
        label = []
        for i in range(self.length):
            if arr[i]:
                label.append(self.array2label[i])
        if len(label) == 1:
            return label[0]
        elif len(label) > 1:
            return label

    def load(self, load_path):
        # 2020-2-13 从指定路径加载步骤
        with open(load_path, 'r') as f:
            self.label2array = json.load(fp = f)
        for key in self.label2array:
            self.array2label[self.label2array[key]] = key
        self.length = len(self.label2array)

    def save(self, save_path):
        # 2020-3-2 将步骤保存至指定路径
        with open(save_path, 'w') as f:
            json.dump(self.label2array, f)

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
                label_dict = os.path.join(package_root, 'labeltemplate',
                                          'aasm.labeltemplate'),
                mode = 'memory'):
        self.VERSION = 0.2
        self.path = save_path
        if self.path is not None and (dataset_name is not None):
            self.set_name(dataset_name)
        self.comment = comment
        self.label_dict = {}
        if isinstance(label_dict, str):
            self.set_label_dict('LABEL', label_dict)
        elif isinstance(label_dict, dict):
            for label in label_dict.keys():
                self.set_label_dict(label, label_dict[label])
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

    def set_label_dict(self, label_name, label_dict):
        self.label_dict[label_name] = LabelDict(label_dict)

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

    def select_epochs(self, label_name=None, label_value=None, data_name=None):
        # data_name
        if data_name is None:
            data_name = self.get_data()
        elif isinstance(data_name, str):
            data_name = [data_name]
        # label_name & label_value
        if (label_name is None) and (label_value is None):
            rst = list()
            for d in data_name:
                if self.__get_state(d) < Dataset.PREPROCESSED:
                    continue
                logging.info(d)
                for epoch in self.get_epochs(d):
                    rst.append((d, epoch))
        elif label_name is not None and label_value is not None:
            rst = list()
            if not isinstance(label_value, list):
                label_value = [label_value]
            for d in data_name:
                if self.__get_state(d) == Dataset.ERROR:
                    continue
                logging.info(d)
                for epoch in self.get_epochs(d):
                    for l in label_value:
                        if self.__get_label(d, epoch, label_name) == l:
                            rst.append((d, epoch))
        else:
            raise TypeError('You should input all of or none of\n' + 
                            '`label_type` and `label_value`')
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
                        if self.__get_label(data_name, epoch, label_name) == l:
                            to_be_del.append(epoch)
                    for e in to_be_del:
                        self.__del_epoch(data_name, e)
        else:
            # When no inputs, all the labels NOT IN LABEL_DICT will be exclude.
            for label_name in self.labels:
                allowed_label = self.label_dict[label_name].labels()
                for data_name in self.get_data():
                    if self.__get_state(data_name) == Dataset.ERROR:
                        continue
                    logging.info(data_name)
                    to_be_del = []
                    for epoch in self.get_epochs(data_name):
                        if self.__get_label(data_name, epoch, label_name) \
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

    def __get_feature(self, data_name, epoch, feature_name):
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
        return self.dataset[data_name]['data'][epoch][feature_name]


    def __df_get_feature(self, disk_file, data_name, epoch, feature_name):
        return disk_file['dataset'][data_name]['data']\
            [str(epoch)][feature_name][()]

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

    def __get_label(self, data_name, epoch, label_name):
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
            label = self.label_dict[label_name].labels()
        elif not isinstance(label, list):
            label = [label]
        for l in label:
            rst[l] = 0
        for dn in data_name:
            if self.__get_state(dn) != Dataset.ERROR:
                logging.info(dn)
                for epoch in self.get_epochs(dn):
                    l = self.__get_label(dn, epoch, label_name)
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
                            self.shape[f] = self.__get_feature(data_name, 
                                                          epoch, f).shape
                        else:
                            tb = 'Shape is inconsistent.'
                            if self.shape[f] != self.__get_feature(data_name, 
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

    def stat_classes(self, elements):
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
        elements : str or list
            `element_name` or a list of them. Must be `label_name` or
            `condition_type`.

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
        
        if isinstance(elements, str):
            elements = [elements]
        elif isinstance(elements, list):
            pass
        else:
            raise TypeError('The input parameter `elements` must be `str` or\n'
                            '`list`. It represents label names or condition\n'
                            'types.')

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
        if len(label) == 0:  # the unit is `data`
            for d in self.get_data():
                c = []
                for cond in condition:
                    c.append(self.get_condition(d, cond))
                c = tuple(c) if num > 1 else c[0]  # use tuple or single
                                                   # element
                if c not in rst:
                    rst[c] = []
                rst[c].append(d)
        else:  # the unit is `epoch`
            for d in self.get_data():
                for e in self.get_epochs(d):
                    c = []
                    for cond in condition:
                        c.append(self.get_condition(d, cond))
                    for l in label:
                        c.append(self.__get_label(d, e, l))
                    c = tuple(c) if num > 1 else c[0]  # use tuple or single 
                                                       # element
                    if c not in rst:
                        rst[c] = []
                    rst[c].append((d, e))
        return rst

    def sample_data(self, data_name, lst, tmin=0, tmax=0, data_padding=True,
                    max_len=None, epoch_padding=False):
        x, y = lst
        x_samp = []
        for i in x:
            x_samp.append(self.sample_serial_x(data_name, i, tmin, tmax,
                                               data_padding, max_len,
                                               epoch_padding))
        if len(x_samp) == 1:
            x_samp = x_samp[0]
        if y is None:
            return x_samp
        else:
            y_samp = []
            for i in y:
                y_samp.append(self.get_condition(data_name, i))
            if len(y_samp) == 1:
                y_samp = y_samp[0]
            return (x_samp, y_samp)

    def sample_epoch(self, data_name, epoch, lst, tmin=0, tmax=0,
                    epoch_padding=False):
        x, y = lst
        x_samp = []
        for i in x:
            x_samp.append(self.sample_epoched_x(data_name, epoch, i,
                                                tmin, tmax, epoch_padding))
        if len(x_samp) == 1:
            x_samp = x_samp[0]
        if y is None:
            return x_samp
        else:
            y_samp = []
            for i in y:
                y_samp.append(self.sample_epoched_y(data_name, epoch, i))
            if len(y_samp) == 1:
                y_samp = y_samp[0]
            return (x_samp, y_samp)

    def sample_epoched_x(self, data_name, epoch, element_name, tmin=0,
                               tmax=0, padding=False, array_type=int):
        '''当返回None时，跳过这个epoch'''
        # check state
        if self.__get_state(data_name) < Dataset.PREPROCESSED:
            raise DataStateError('You cannot sample from a data not correctly '
                                 'preprocessed. The target data `' + data_name
                                 + '` has a state `' 
                                 + self.__get_state(data_name, True) + '`.')
        # without timestep
        if tmin == 0 and tmax == 0:
            # feature
            if self.elements[element_name] == 'feature':
                return self.__get_feature(data_name, epoch, element_name)
            # label
            elif self.elements[element_name] == 'label':
                return self.__get_label(data_name, epoch, element_name)
        # with timestep
        else:
            # check timespan
            # timespan cannot be longer than data_length
            data_length = len(self.get_epochs(data_name))
            logging.info('Data Length: ' + str(data_length))
            assert tmax >= 0
            assert abs(tmin) + tmax <= data_length
            # sample
            epoch_list = self.get_epochs(data_name)
            # at the start of the series
            if epoch_list[epoch] < epoch_list[abs(tmin)]:
                if padding:  # pre-padding
                    # feature
                    if self.elements[element_name] == 'feature':
                        r = np.array([np.zeros(self.shape[element_name])] \
                                     * (abs(tmin) - epoch) 
                            + [self.__get_feature(data_name, i, element_name) \
                               for i in epoch_list[: epoch + tmax + 1]])
                    # label
                    elif self.elements[element_name] == 'label':
                        p = self.label_dict[element_name].get_array( \
                                                None, array_type = array_type)
                        r = np.array([p] * (abs(tmin) - epoch) 
                            + [self.label_dict[element_name].get_array( \
                            self.__get_label(data_name,
                                             epoch[i], element_name), \
                                array_type = array_type) \
                               for i in epoch_list[: epoch + tmax + 1]])
                # if not padding, the epoch at edge will be disposed
                else:
                    raise BrokenTimestepError()
            # at the end of the series
            elif epoch_list[epoch] > epoch_list[data_length - tmax - 1]:
                if padding:  # post-padding
                    # feature
                    if self.elements[element_name] == 'feature':
                        r = np.array([self.__get_feature(data_name,
                                                         i, element_name) \
                                      for i in epoch_list[epoch - abs(tmin) :]]
                        + [np.zeros(self.shape[element_name])] \
                            * (tmax + epoch + 1 - data_length))
                    # label
                    elif self.elements[element_name] == 'label':
                        p = self.label_dict[element_name].get_array( \
                                                None, array_type = array_type)
                        r = np.array([self.label_dict[element_name].get_array(
                                        self.__get_label(data_name,
                                                         i, element_name), \
                                            array_type = array_type) \
                                      for i in epoch_list[epoch - abs(tmin) :]]
                        + [p] * (tmax + epoch + 1 - data_length))
            else: # normal situation
                # feature
                if self.elements[element_name] == 'feature':
                    r = np.array([self.__get_feature(data_name, i,
                                                     element_name) 
                                  for i in epoch_list[epoch - abs(tmin) : \
                                                      epoch + tmax + 1]])
                # label
                elif self.elements[element_name] == 'label':
                    r = np.array([self.label_dict[element_name].get_array( \
                            self.__get_label(data_name, epoch[i], element_name),
                            array_type = array_type)
                            for i in epoch_list[epoch - abs(tmin) : \
                                                      epoch + tmax + 1]])
            return r

    def sample_epoched_y(self, data_name, epoch, element_name, array_type=int):
        # check state
        if self.__get_state(data_name) < Dataset.PREPROCESSED:
            raise DataStateError('You cannot sample from a data not correctly '
                                 'preprocessed. The target data `' + data_name
                                 + '` has a state `' 
                                 + self.__get_state(data_name, True) + '`.')
        # label
        if self.elements[element_name] == 'label':
            return self.__get_label(data_name, epoch, element_name)
        elif self.elements[element_name] == 'condition':
            return self.get_condition(data_name, condition_type=element_name)
        else:
            raise ValueError('The input element ' + element_name + 'is a `'
                             + self.elements[element_name] + '`. It cannot'
                             'act as `epoched_y` sample.')
    
    def sample_serial_x(self, data_name, element_name, tmin, tmax,
                        data_padding, max_len, epoch_padding):
        rst = []
        for epoch in self.get_epochs(data_name):
            try:
                rst.append(self.sample_epoched_x(data_name, epoch, 
                                                 element_name,
                                                 tmin, tmax, epoch_padding))
            except BrokenTimestepError:
                continue
        if data_padding:
            if max_len is None:
                raise ValueError('A certain `max_len` is needed for '
                                 'data_padding')
            if self.elements[element_name] == 'feature':
                sample_shape = self.shape[element_name]
                dtype = 'float32'
            else:
                sample_shape = self.label_dict[element_name].shape()
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
            for label_name in self.label_dict.keys():
                disk_file['label_dict'].create_group(label_name)
                tem = self.label_dict[label_name].label2array
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
                                      self.__get_feature(data_name, epoch, feature))
            # 保存标签 2020-3-5
            if self.has_label:
                for label in self.labels:
                    self.__df_set_label(disk_file, data_name, epoch, label, 
                               self.__get_label(data_name, epoch, label))

    def df_load_label_dict_0_2(self, disk_file):
        # 获取标签字典
        if 'label_dict' in disk_file.keys():
            for label_name in disk_file['label_dict'].keys():
                self.label_dict[label_name] = LabelDict()
                for label in disk_file['label_dict'][label_name].keys():
                    self.label_dict[label_name].add_label(label, order = int(disk_file['label_dict'][label_name][label][()])) # 2020-3-5 存的是int

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


class IndexManager(object):
    def __init__(self, *args):
        self.idx = {}
        