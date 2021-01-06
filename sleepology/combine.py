# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:23:26 2020

@author: 赵匡是
"""

import glob
from . import dataset

files = glob.glob('/userhome/result/shhs_psd/shhs_psd_*.dataset')

ds0 = dataset.Dataset()
ds0.load(files.pop())

for f in files:
    ds1 = dataset.Dataset()
    ds1.load(f)
    ds0.dataset.update(ds1.dataset)

ds0.save('shhs_PSD_combined')
