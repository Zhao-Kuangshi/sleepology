# -*- coding: utf-8 -*-
import os
import json

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