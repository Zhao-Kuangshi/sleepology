# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:02:45 2020

@author: Zhao Kuangshi
"""

from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
from typing import Sequence, Dict
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers:Dict = {}, verbose:bool = False):
    '''
    Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    '''
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def is_one_hot(array: Sequence) -> bool:
    '''
    Check if an array is a one-hot array.

    Parameters
    ----------
    array : Sequence
        The array to be checked..

    '''
    return sum([bool(i) for i in array]) == 1

def argmax(array: Sequence) -> int:
    idx = 0
    m = array[0]
    for i, v in enumerate(array):
        if v > m:
            idx = i
            m = v
    return idx
