# -*- coding: utf-8 -*-
import json
import numpy as np
from typing import List, Tuple, Dict, Union, Sequence

class BaseDict(object):
    pass



class GradeDict(BaseDict):
    pass


class FloatDict(BaseDict):
    pass
    
    


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
    #  2020-3-2 标签字典
    def __init__(self, load_path = None):
        # 2020-3-2 array存的是数字
        self.dict = {}
        self.reverse_dict = {}
        self.label_type = None
        self.value_type = int
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

    def __init__(self, content=None):
        # 2020-3-2 array存的是数字
        self.dict = {}
        self.label_type = None
        self.value_type = None
        
        self.length = 0 # 2020-3-2 提供当前字典的长度信息，也是one_hot数组的长度参考
        if isinstance(content, list):
            for i in content:
                self.add_label(i)
        elif isinstance(content, dict):
            self.dict = content
        elif isinstance(content, str):
            self.load(content)

    def array_mode(self) -> bool:
        '''
        The mode discriminate whether the label dictionary can translate one
        label (the key of the dict) into an array. If values of labels are
        `int` or `list`, absolutely they can be translated into an array.
        Otherwise if values are `float`, the label willl be recognized as a
        regression target.
        '''
        pass
        

    def shape(self, N_hot=True):
        if N_hot:
            return (self.length, )
        else:
            return (1,)

    def labels(self):
        return list(self.label2array.keys())

    def add_label(self,
                  label: Union[LabelType, Dict[LabelType, ValueType]],
                  value: Union[ValueType, None] = None) -> None:
        # There are three ways to add labels into `LabelDict`:
        # 1. you can input a `dict` with labels as keys and values as values;
        # 2. you can input only a label if you want to generate a value
        # automatically;
        # 3. you can input a label and a value so that you can define the value
        # of new label.
        # TODO : 写输入dict的情况 &&&&& 补充key的类型检查
        
        # === label ===
        if isinstance(label, dict):
            # check if the label type accord with the previous label
        # === value ===
        if value is None:  # no value input
            # check if other values are int
            value = len(self.dict)
        # check whether the value is in correct type
        elif not isinstance(value, Union[int, List[int, bool], float]):
            raise TypeError('cannot understand the input value of label. '
                            'Expect `int`, list of `int`, list of `bool` or '
                            '`float`, but receive {0}.'.format(type(value)))
        if isinstance(value, List[int, bool]) and \
            sum([bool(i) for i in value]) != 1:
            # ensure there is only one `1` or `True` in the array and others
            # are `0` or `False`, i.e., ensuring the value array is `one-hot`
            # array
            raise ValueError('only one-hot array is valid.')
        # add new label
        self.dict[label] = value
        self.length += 1

    def add_default(self, default_value):
        self.default_value = default_value

    def __add_label_by_dict(self, d):
        pass
        # check if the types of the input labels are match
        
        # check if the types of the input values are match
        
        # add to dict
    
    def __add_label_by_label(self, l):
        # check if the types of value of the existing dict, i.e., `self.dict`,
        # is integer. If not, raise an Error
        
        # check if the types of the input labels are match
        
        pass
    
    def __add_label_by_label_and_value(self, l, v):
        pass
    
    def __label_check(self, label: LabelType) -> None:
        '''
        Check if the type of the new label accords with the previous.

        Parameters
        ----------
        label : LabelType
            The label to be checked.

        Raises
        ------
        TypeError
            Raised when the new label does not accord with the previous.

        '''
        if self.label_type is None:  # the first label to add
            self.label_type = type(label)
        elif self.label_type == type(label):
            return None
        else:
            raise TypeError('the type of the new label `{0}` does not accord '
                            'with previous labels whose type is `{1}`'.format(
                                type(label), self.label_type))

    def __value_check(self, value) -> None:
        # when the value is a sequence
        if isinstance(value, list):
            # check if the sequence is one-hot
            if not self.__is_one_hot(value):
                raise ValueError('you must add a one-hot array to the '
                                 'LabelDict')
            # and then translate sequential array into a number
            value = np.argmax(value)
        if self.value_type is None:  # the first value to add
            self.value_type = type(value)
        elif self.value_type == type(value):
            return None
        else:
            raise TypeError('the type of the new value `{0}` does not accord '
                            'with previous values whose type is `{1}`'.format(
                                type(value), self.value_type))
            

    def __is_one_hot(self, array: Sequence) -> bool:
        return sum([bool(i) for i in array]) != 1

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

    def load(self, load_path):
        # 2020-2-13 从指定路径加载步骤
        with open(load_path, 'r') as f:
            self.dict = json.load(fp = f)
        self.length = len(self.dict)

    def save(self, save_path):
        # 2020-3-2 将步骤保存至指定路径
        with open(save_path, 'w') as f:
            json.dump(self.dict, f)