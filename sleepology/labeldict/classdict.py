 # -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:28:23 2021

@author: Zhao Kuangshi
"""

import os
import json
import numpy as np
from collections.abc import Iterable
from typing import List, Tuple, Dict, Union, Sequence, Optional

from ..utils import is_one_hot, argmax
from ..exceptions import LabelValueConflictError

from .basedict import BaseDict

# Annote the types of labels and values
LabelType = Union[int,
                  float,
                  str,
                  Tuple[int, int],
                  Tuple[float, float]]
ValueType = Union[int,
                  List[int],
                  List[bool],
                  float]
type_str = {
    int: 'int',
    float: 'float',
    str: 'str',
    tuple: 'tuple'
    }


class ClassDict(BaseDict):
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
    def __init__(self, *content):
        self.dict_type = 'ClassDict'
        self.dict = {}
        self.reverse_dict = {}
        self.label_type = None
        self.value_type = int
        self.length = 0 # provide the length of current dict
        if len(content) == 1 and isinstance(content[0], str) and \
            os.path.exists(content[0]):
            self.load(content[0])
        elif len(content) > 0:
            self.add(*content)

    def shape(self, N_hot=True):
        if N_hot:
            return (self.length, )
        else:
            return (1,)

    def labels(self) -> list:
        return list(self.dict.keys())

    def __value(self) -> list:
        return sorted(list(self.reverse_dict.keys()))
    
    def add(self, *content):
        if len(content) == 1 and isinstance(content[0], dict):
            self.add_by_dict(content[0])
        elif len(content) == 1 and isinstance(content[0], list):
            for l in content[0]:
                self.add_by_label(l)
        elif len(content) == 1:
            self.add_by_label(content[0])
        elif len(content) == 2:
            self.add_by_label(content[0], content[1])
        elif len(content) == 0:
            raise TypeError('you must give some parameters.')
        else:
            raise TypeError('too much parameters.')

    def add_by_dict(self, d: Dict) -> None:
        # check keys
        for k in d:
            self.add_by_label(k, d[k])

    def add_by_label(self, label: LabelType, value: ValueType = None) -> None:
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

    def trans(self, label: Union[LabelType, List[LabelType]],
              array_type: type = int,
              dense: bool = False):
        if self.length <= 2:
            return self.get_number(label, dense)
        else:
            return self.get_array(label, array_type, dense)

    def get_value(self, label: Optional[LabelType], dense: bool = False):
        if isinstance(label, list):
            if len(label) != 1:
                raise TypeError(
                    'make sure your request is single-label when array_type is'
                    ' None.')
        elif label is None:  # request padding
            return -1  # `-1` represents padding
        elif dense:
            return self.__value().index(self.dict[label])
        else:
            return self.dict[label]

    def get_array(self, 
                  label: Union[LabelType, List[LabelType], None],
                  array_type: type = int,
                  dense: bool = False):
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
            self.get_number(label)
        # return N-hot array
        else:
            if isinstance(label, list) and len(label) == 0:
                raise TypeError('at least 1 label should be given.')
            if label is None:  # request padding
                arr = np.asarray([False] * self.length)
                return arr.astype(array_type)
            elif dense:
                val = self.__value()
                if isinstance(label, str):
                    label = [label]
                arr = np.asarray([False] * self.length)
                for l in label:
                    arr[val.index(self.dict[l])] = True
                return arr.astype(array_type)
            else:
                if isinstance(label, str):
                    label = [label]
                arr = np.asarray(
                    [False] * (max(list(self.reverse_dict.keys())) + 1))
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

    def save(self, save_path: str) -> None:
        '''
        Save the LabelDict.

        Parameters
        ----------
        save_path : str
            The target for saving LabelDict.
        '''
        save_path = os.path.expenduser(save_path)
        struct = {"dict type": self.dict_type,
                  "dict": self.dict,
                  "reverse_dict": self.reverse_dict,
                  "label_type": type_str[self.label_type],
                  "value_type": type_str[self.value_type],
                  "length": self.length}
        with open(save_path, 'w') as f:
            json.dump(struct , f)

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
                if 'alias' in self.reverse_dict[key] and \
                    primary in self.reverse_dict[key]['alias']:
                    self.reverse_dict[key]['alias'].remove(primary)
                else:
                    self.reverse_dict[key]['alias'] = []
                self.reverse_dict[key]['alias'].extend(
                        self.reverse_dict[key]['primary'])
                self.reverse_dict[key]['primary'] = primary
        else:
            self.reverse_add(key, primary)
            

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
            return label
        elif self.label_type == type(label) and label not in self.dict:
            return label
        elif self.label_type == type(label) and label in self.dict:
            raise ValueError(f'the label `{label}` has already existed in the '
                             'LabelDict. If you want to modify the value of '
                             f'{label}, please use `LabelDict.modify()`.')
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
            if not is_one_hot(value):
                raise ValueError('you must add a one-hot array to the '
                                 'LabelDict')
            # and then translate sequential array into a number
            value = argmax(value)
        if self.value_type is None:  # the first value to add
            self.value_type = type(value)
            return value
        elif self.value_type == type(value):
            return value
        else:
            raise TypeError('the type of the new value `{0}` does not accord '
                            'with previous values whose type is `{1}`'.format(
                                type(value), self.value_type))



