#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 19:15:36 2021

@author: zhaokuangshi
"""

from .basedict import BaseDict
from .classdict import ClassDict
from .gradedict import GradeDict
from .valuedict import ValueDict

class AASM(ClassDict):
    def __init__(self, *content):
        super().__init__(['W', '1', '2', '3', 'R'])

class AASMIncludeMovement(ClassDict):
    def __init__(self, *content):
        super().__init__(['W', '1', '2', '3', 'R', 'M'])

class RK(ClassDict):
    def __init__(self, *content):
        super().__init__(['W', '1', '2', '3', '4', 'R'])