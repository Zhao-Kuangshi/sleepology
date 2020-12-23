# -*- coding: utf-8 -*-
import json
import numpy as np
from collections.abc import Iterable


class LabelDict(object):
    #  2020-3-2 标签字典
    def __init__(self, load_path = None):
        # 2020-3-2 array存的是数字
        self.label2array = {}
        self.array2label = {}
        self.length = 0 # 2020-3-2 提供当前字典的长度信息，也是one_hot数组的长度参考
        if load_path is not None:
            self.load(load_path)

    def shape(self, N_hot=True):
        if N_hot:
            return (self.length, )
        else:
            return (1,)

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
                return self.label2array[label]
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