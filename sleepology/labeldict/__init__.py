 # -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:15:59 2021

@author: Zhao Kuangshi
"""
import os
import json

from .basedict import BaseDict
from .classdict import ClassDict
from .gradedict import GradeDict
from .valuedict import ValueDict
from .utils import *


str_type = {
    'int': int,
    'float': float,
    'str': str,
    'tuple': tuple
    }

def load(fpath: str) -> BaseDict:
    '''
    Load the external, customerized LabelDict from file system.

    Parameters
    ----------
    fpath : str
        The file to be loaded.

    Raises
    ------
    ValueError
        Raised when the file cannot be resolved by the function.

    Returns
    -------
    BaseDict

    '''
    try:
        fpath = os.path.expanduser(fpath)
        with open(fpath, 'r') as f:
            struct = json.load(f)
        sub = struct['dict type']
    except:
        raise ValueError('wrong file or broken file.')
    if sub == 'ClassDict':
        return __load_classdict(struct)
    elif sub == 'GradeDict':
        return __load_gradedict(struct)
    elif sub == 'ValueDict':
        return __load_valuedict(struct)
    else:
        raise ValueError('unknown LabelDict type. It may because the file is '
                         'broken, or you are using a old version of sleepology'
                         ' but loading a newer version of LabelDict.')

def __load_classdict(struct: dict) -> ClassDict:
    '''
    A function to load and generate a ClassDict.

    Parameters
    ----------
    struct : dict
        A dict read from JSON file.

    Returns
    -------
    ClassDict

    '''
    # `value_type` of ClassDict must be int
    assert struct['value_type'] == 'int'
    ret = ClassDict()
    ret.label_type = str_type[struct['label_type']]
    ret.dict = struct['dict']
    reverse_dict = {}
    for i in struct['reverse_dict']:
        reverse_dict[int(i)] = struct['reverse_dict'][i]
    ret.reverse_dict = reverse_dict
    ret.length = struct['length']
    return ret
    

def __load_gradedict(struct: dict) -> GradeDict:
    '''
    A function to load and generate a GradeDict.

    Parameters
    ----------
    struct : dict
        A dict read from JSON file.

    Returns
    -------
    GradeDict

    '''
    pass

def __load_valuedict(struct: dict) -> ValueDict:
    '''
    A function to load and generate a ValueDict.

    Parameters
    ----------
    struct : dict
        A dict read from JSON file.

    Returns
    -------
    ValueDict

    '''
    pass