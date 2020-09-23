# -*- coding: utf-8 -*-
import os
import datetime


class ExceptionLogger(object):
    def __init__(self, dataset):
        self.create = False
        self.dataset = dataset

    def first_line(self):
        time = datetime.datetime.now()
        postfix = time.strftime('%Y%m%d%H%M%S')
        fname = 'log_' + postfix + '.log'
        self.dst = os.path.join(self.dataset.path, fname)
        print('The log will be saved at')
        print(self.dst)
        wrapper = '\n'
        line = '============================================================='
        self.create = True
        with open(self.dst, 'w') as f:
            f.write('ERROR LOG' + wrapper)
            f.write(line + wrapper)
            f.write(wrapper)
            f.write('Dataset Name: ' + self.dataset.name + wrapper)
            f.write('Time: ' + str(time) + wrapper)
            f.write(line + wrapper)

    def submit(self, data_name, epoch, feature_or_label_name, trace_back):
        if not self.create:
            self.first_line()
        wrapper = '\n'
        tab = '\t'
        line = '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
        with open(self.dst, 'a') as f:
            f.write(wrapper)
            f.write(wrapper)
            f.write('DATA: ' + data_name + wrapper)
            if epoch is not None:
                f.write('EPOCH: ' + str(epoch) + tab)
            if feature_or_label_name is not None:
                f.write('SERIES: ' + feature_or_label_name + wrapper)
            f.write(trace_back + wrapper)
            f.write(line + wrapper)