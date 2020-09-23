# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:22:57 2020

@author: 赵匡是
"""
import os
import re
import mne
import math
import numpy
import pylsl
import traceback

mne.set_log_level('WARNING')


class Source(object):
    '''
    要有获取数据的方法
    '''
    def __init__(self, path, source_type):
        self.path = path
        self.source_type = source_type

    def get(self):
        '''
        An abstract method to get next raw data epoch.

        Raises
        ------
        NotImplementedError

        Returns
        -------
        None.

        '''
        raise NotImplementedError('Please implement the method.')


class BCISource(Source):
    def __init__(self, path, source_type, 
                 epoch_length, sample_freq):
        super().__init__(path, source_type)
        self.epoch_length = epoch_length
        self.sample_freq = sample_freq

    


class RawDataFile(BCISource):
    def __init__(self, path, source_type, 
                 epoch_length, sample_freq, name = None):
        super().__init__(path, source_type, 
                         epoch_length, sample_freq)
        if name is None:
            self.name = os.path.basename(self.path)
        else:
            self.name = name

    def get(self, number = None):
        '''
        Get the data of one epoch. The length (time span) of each epoch equals 
        to `epoch_length` you have set when initialize this object.
        You can specify which epoch you want to get by setting parameter 
        `number`.
        When `number is None`, this method will sequentially return epochs 
        until meet the end of data. Then function will raise a `StopIteration`
        exception.
        
        NOTE: After initializing the `RawDataFile` object, when first time 
        using `get` method, it will load data, resample data and split data.
        So the first usage will spend a long time.

        Parameters
        ----------
        number : int, optional
            The serial number of the epoch. You can specify which epoch you 
            want to get. The default is None, means sequentially return epochs 
            until meet the end of data.

        Raises
        ------
        StopIteration
            Encounter the end of data. You may use `try ... except` structure 
            to handle.

        Returns
        -------
        number : int
            The serial number of the epoch.
        epoch : mne.epochs.Epochs

        '''
        if not hasattr(self, 'raw'):
            self.__load_data()
            self.number = 0
        try:
            if number is None:
                number = self.number
                self.number += 1
            return number, self.raw[number]
        except IndexError:
            raise StopIteration
        except AttributeError:
            if not hasattr(self, 'number'):
                raise StopIteration


    def clean(self):
        '''
        Delete the data from memory.

        '''
        del self.raw

    def __load_data(self, args = ()):
        '''
        A private method for load data, resample data and split data into 
        epochs.

        Parameters
        ----------
        args : tuple, optional
            The additional parameters for reading data. The default is ().

        '''
        self.raw = self.__read_raw_data(self.path, self.source_type, args)
        self.raw.load_data()
        self.raw.resample(self.sample_freq)
        events = numpy.array([[i, 0, 1] for i in range(0, self.raw.n_times, int(self.epoch_length * self.raw.info['sfreq']))])
        self.raw = mne.Epochs(self.raw, events, tmin = 0, tmax = self.epoch_length, baseline = None)

    def __read_raw_data(self, fpath, typ, args = ()):
        '''
        A private method for openning raw BCI data files.

        Parameters
        ----------
        fpath : str
            The file path of the raw data file.
        typ : str
            The type of a raw data file. The available type is shown below. 
            You can use the complete type name or the extension in the bracket.
            - BrainVision (vhdr)
            - European data format (edf)
            - BioSemi data format (bdf)
            - General data format (gdf)
            - Neuroscan CNT data format (cnt)
            - EGI simple binary (egi)
            - EGI MFF (mff)
            - EEGLAB set files (set)
            - Nicolet (data)
            - eXimia EEG data (nxe)
        args : tuple, optional
            Additional arguments. Which is depend on file type. You can get 
            detailed usage on 
            https://mne.tools/stable/python_reference.html#reading-raw-data . 
            The default is ().

        Returns
        -------
        mne.io.Raw
        '''
        # 2020-2-21 导入原始数据
        if typ == 'vhdr' or typ == 'BrainVision':
            return mne.io.read_raw_brainvision(fpath, *args)
        elif typ == 'edf' or typ == 'European data format':
            return mne.io.read_raw_edf(fpath, *args) 
        elif typ == 'bdf' or typ == 'BioSemi data format':
            return mne.io.read_raw_bdf(fpath, *args) 
        elif typ == 'gdf' or typ == 'General data format':
            return mne.io.read_raw_gdf(fpath, *args) 
        elif typ == 'cnt' or typ == 'Neuroscan CNT data format':
            return mne.io.read_raw_cnt(fpath, *args) 
        elif typ == 'egi' or typ == 'EGI simple binary':
            return mne.io.read_raw_egi(fpath, *args) 
        elif typ == 'mff' or typ == 'EGI MFF':
            return mne.io.read_raw_egi(fpath, *args) 
        elif typ == 'set' or typ == 'EEGLAB set files':
            return mne.io.read_raw_eeglab(fpath, *args) 
        elif typ == 'data' or typ == 'Nicolet':
            return mne.io.read_raw_nicolet(fpath, *args) 
        elif typ == 'nxe' or typ == 'eXimia EEG data':
            return mne.io.read_raw_eximia(fpath, *args)        



class TCPStream(BCISource):
    # 增加一个self.cursor参数，在get数据时，同步修改cursor。这样可以确定数据的顺序。
    pass


class NihonkohdenLabelSource(Source):
    def __init__(self, path, source_type = None):
        super().__init__(path, source_type)

    def get(self, number = None):
        '''
        Get the label of one epoch. 
        You can specify which epoch you want to get by setting parameter 
        `number`.
        When `number is None`, this method will sequentially return epochs 
        until meet the end of data. Then function will raise a `StopIteration`
        exception.

        Parameters
        ----------
        number : int, optional
            The serial number of the epoch. You can specify which epoch you 
            want to get. The default is None, means sequentially return epochs 
            until meet the end of data.

        Raises
        ------
        StopIteration
            Encounter the end of data. You may use `try ... except` structure 
            to handle.

        Returns
        -------
        number : int
            The serial number of the epoch.
        epoch : mne.epochs.Epochs

        '''
        if not hasattr(self, 'raw'):
            self.__load_data()
            self.number = 0
        try:
            if number is None:
                number = self.number
                self.number += 1
            rst = self.raw[number]
            if self.source_type == 'aasm':
                rst = self.trans_aasm(rst)
            return number, rst
        except IndexError:
            raise StopIteration

    def trans_aasm(self, label):
    # 2020-3-2 把标签中的4期睡眠全部转化为3期
        if label == '4':
            label = '3'
        return label

    def clean(self):
        '''
        Delete the data from memory.

        '''
        del self.raw

    def __load_data(self):
        with open(self.path, 'r') as f:
            label_raw = f.readlines()     
        self.raw = []
        for entry in label_raw:
            self.raw.append(re.split('\\s+', entry)[-3]) 

    