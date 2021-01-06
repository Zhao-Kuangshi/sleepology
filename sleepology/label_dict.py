# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from collections.abc import Iterable
from typing import List, Tuple, Dict, Union, Sequence
from .utils import is_one_hot, argmax
from .exceptions import LabelValueConflictError


class ClassDict(object):
    '''
    Label dictionary for classification.

    Mainly used to convert original category labels into integer numbers
    suitable for machine learning (for statistical machine learning, for
    example scikit-learn) OR into one-hot /N-hot vectors (for machine learning
    methods using activation functions such as softmax/sigmoid, usually deep
    learning).

    One dictionary can be used in different datasets (i.e., different `Dataset`
    instances) and sampling schemes (i.e., different `Sample` instances), which
    can make different datasets have consistent label translation for machine
    learning models. The translation of labels will be based entirely on the
    definition of dictionaries, so as long as the same label dictionary is
    used, even if the new batch of data lacks one or more labels, or the order
    of labels is different. It will not affect the final label classification
    results.
    '''
    # Annote the types of labels and values
    LabelType = Union[int,
                      float,
                      str,
                      Tuple[int, int],
                      Tuple[float, float],
                      List[str]]
    ValueType = Union[int,
                      List[int],
                      List[bool],
                      float]
    def __init__(self, *content):
        self.dict = {}
        self.reverse_dict = {}
        self.label_type = None
        self.value_type = int
        self.length = 0 # 2020-3-2 提供当前字典的长度信息，也是one_hot数组的长度参考
        if len(content) == 1 and os.path.exists(content[0]):
            self.load(content[0])
        elif len(content) > 0:
            self.add(content)

    def shape(self, N_hot=True):
        if N_hot:
            return (self.length, )
        else:
            return (1,)

    def labels(self):
        return list(self.dict.keys())
    
    def add(self, *content):
        if len(content) == 1 and isinstance(content[0], dict):
            self.add_by_dict(content[0])
        elif len(content) == 1:
            self.add_by_label(content[0])
        elif len(content) == 2:
            self.add_by_label(content[0], content[1])

    def add_by_dict(self, d: Dict) -> None:
        # check keys
        for k in d:
            self.__label_check(k)
            d[k] = self.__value_check(d[k])
        self.dict.update(d)
        for k in d:
            self.reverse_add(d[k], k)

    # TODO: todo
    def add_by_label(self, label, value = None) -> None:
        # 2020-3-2
        if value is None and len(self.dict) > 0:
            value = max(list(self.dict.values())) + 1
        elif value is None:
            value = 0
        label = self.__label_check(label)
        value = self.__value_check(value)
        self.dict[label] = value
        self.reverse_add(value, label)
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
                array_type='bool'        (Wrong)

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
            if isinstance(label, list):
                if len(label) != 1:
                    raise TypeError(
                        'make sure your request is single-label when'
                        ' array_type is None.')
            elif label is None:  # request padding
                return -1  # `-1` represents padding
            else:
                return self.dict[label]
        # return N-hot array
        else:
            if isinstance(label, list) and len(label) == 0:
                raise TypeError('at least 1 label should be given.')
            if label is None:  # request padding
                arr = np.asarray([False] * self.length)
                return arr.astype(array_type)
            else:
                if isinstance(label, str):
                    label = [label]
                arr = np.asarray([False] * self.length)
                for l in label:
                    arr[self.dict[l]] = True
                return arr.astype(array_type)

    def get_label(self, array):
        if len(array) != self.length:
            raise TypeError('array length dismatch')
        arr = array.astype(bool)
        label = []
        for i in range(self.length):
            if arr[i]:
                label.append(self.reverse_dict[i])
        if len(label) == 1:
            return label[0]
        elif len(label) > 1:
            return label

    def load(self, load_path):
        # 2020-2-13 从指定路径加载步骤
        with open(load_path, 'r') as f:
            self.dict = json.load(fp = f)
        for key in self.dict:
            self.reverse_dict[self.dict[key]] = key
        self.length = len(self.dict)

    def save(self, save_path):
        # 2020-3-2 将步骤保存至指定路径
        with open(save_path, 'w') as f:
            json.dump(self.dict, f)

    def reverse_add(self, key: ValueType, value: LabelType,
                    alias: bool = False) -> None:
        if key in self.reverse_dict:
            if not alias:
                raise LabelValueConflictError(
                    'the label \'{0}\' has the same value as \'{1}\', but '
                    'only one primary label could be added. If you regard '
                    '\'{0}\' as an alias, please set `alias=True`. If you '
                    'think \'{0}\' is a primary label, use '
                    '`LabelDict.reverse_set_primary()`.'.format(value,
                                          self.reverse_dict[key]['primary']))
            elif 'alias' not in self.reverse_dict[key]:
                self.reverse_dict[key]['alias'] = [value]
            else:
                self.reverse_dict[key]['alias'].extend(value)
        else:
            self.reverse_dict[key] = dict()
            self.reverse_dict[key]['primary'] = value

    def reverse_set_primary(self, key: ValueType, primary: LabelType) -> None:
        if key in self.reverse_dict:
            if self.reverse_dict[key]['primary'] != primary:
                if primary in self.reverse_dict[key]['alias']:
                    self.reverse_dict[key]['alias'].remove(primary)
                self.reverse_dict[key]['alias'].extend(
                    self.reverse_dict[key]['primary'])
                self.reverse_dict[key]['primary'] = primary
                    
            

    def __label_check(self, label: LabelType) -> LabelType:
        '''
        Check if the type of the new label accords with the previous. And check
        if there has been a same label in the LabelDict.

        Parameters
        ----------
        label : LabelType
            The label to be checked.

        Raises
        ------
        ValueError
            Raised when the label has already existed.
        TypeError
            Raised when the new label does not accord with the previous.

        '''
        if self.label_type is None:  # the first label to add
            self.label_type = type(label)
        elif self.label_type == type(label) and label not in self.dict:
            return label
        elif self.label_type == type(label) and label in self.dict:
            raise ValueError(f'the label `{label}` has already existed in the '
                             'LabelDict. If you want to modify the value of '
                             '{label}, please use `LabelDict.modify()`.')
        else:
            raise TypeError('the type of the new label `{0}` does not accord '
                            'with previous labels whose type is `{1}`'.format(
                                type(label), self.label_type))

    def __value_check(self, value: ValueType) -> ValueType:
        '''
        Check if the type of the new value accords with the previous. And check
        if the value is valid.

        Parameters
        ----------
        value : ValueType
            The value to be checked.

        Raises
        ------
        ValueError
            Raised when receiving an array but it is not one-hotted.
        TypeError
            Raised when the new value does not accord with the previous.

        '''
        # when the value is a sequence
        if isinstance(value, list):
            # check if the sequence is one-hot
            if not self.__is_one_hot(value):
                raise ValueError('you must add a one-hot array to the '
                                 'LabelDict')
            # and then translate sequential array into a number
            value = argmax(value)
        if self.value_type is None:  # the first value to add
            self.value_type = type(value)
        elif self.value_type == type(value):
            return value
        else:
            raise TypeError('the type of the new value `{0}` does not accord '
                            'with previous values whose type is `{1}`'.format(
                                type(value), self.value_type))



