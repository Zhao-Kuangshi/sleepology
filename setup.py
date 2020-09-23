# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='sleepology',
    version='0.2.22',
    author = 'Zhao-Kuangshi',
    author_email = 'contact@zhaokuangshi.cn',
    maintainer = 'Zhao-Kuangshi',
    maintainer_email = 'contact@zhaokuangshi.cn',
    license = 'GPL-3.0 License',
    description = 'A python package to manage, process and sample huge ' \
                  'polysomnogram (PSG) datasets. And it offers functions for '\
                  'machine learning (ML), real-time sampling and closed-loop'\
                  ' stimulation in the future. ',
    #packages = ['sleepology'],
    #package_dir = {'sleepology': 'sleepolgy'},
    py_modules = ['sleepology.__init__',
                  'sleepology.dataset',
                  'sleepology.exception_logger',
                  'sleepology.exceptions',
                  'sleepology.label_dict',
                  'sleepology.preprocess',
                  'sleepology.sample',
                  'sleepology.source',
                  'sleepology.utils'],
    data_files = [('labeltemplate', ['labeltemplate/aasm.labeltemplate'])],
    python_requires = '>=3.5',
    install_requires = ['h5py>=2.10.0',
                        'numpy>=1.16.4',
                        'matplotlib',
                        'datetime',
                        'mne>=0.18.1',
                        'tableprint>=0.9.1',
                        'sklearn>=0.19.2',
                        'pylsl']
)