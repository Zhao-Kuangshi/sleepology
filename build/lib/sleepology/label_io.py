# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:37:54 2020

@author: chizh

全部是，输入文件，输出以帧为单位的label list
"""
import re

def trans_aasm(label):
    # 2020-3-2 把标签中的4期睡眠全部转化为3期
    return ['3' if l == '4' else l for l in label]

def read_nihonkohden_txt(label_file, *epoch_length):
    # 2020-3-2 这里epoch_length没有用
    label_raw = open(label_file).readlines()
    label = []
    for entry in label_raw:
        label.append(re.split('\\s+', entry)[-3]) 
    return label

def read_shhs_xml(label_file, *epoch_length):
    label_dict = {0 : 'W',
                  1 : '1',
                  2 : '2',
                  3 : '3',
                  4 : '4',
                  5 : 'R'}
    with open(label_file, 'r') as f:
        content = f.read()
    # Check that there is only one 'Start time' and that it is 0
    patterns_start = re.findall(
        r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', 
        content)
    assert len(patterns_start) == 1
    # Now decode file: find all occurences of EventTypes marking sleep stage annotations
    patterns_stages = re.findall(
        r'<EventType>Stages.Stages</EventType>\n' +
        r'<EventConcept>.+</EventConcept>\n' +
        r'<Start>[0-9\.]+</Start>\n' +
        r'<Duration>[0-9\.]+</Duration>', 
        content)
    print(patterns_stages[-1])
    stages = []
    starts = []
    durations = []
    for pattern in patterns_stages:
        lines = pattern.splitlines()
        stageline = lines[1]
        stage = int(stageline[-16])
        startline = lines[2]
        start = float(startline[7:-8])
        durationline = lines[3]
        duration = float(durationline[10:-11])
        assert duration % 30 == 0.
        epochs_duration = int(duration) // 30

        try:
            stages += [label_dict[stage]]*epochs_duration
        except KeyError:
            stages += ['L']*epochs_duration # 2020-5-15 未知的label则为L
        finally:
            starts += [start]
            durations += [duration]
    # last 'start' and 'duration' are still in mem
    # verify that we have not missed stuff..
    assert int((start + duration)/30) == len(stages)
    return stages