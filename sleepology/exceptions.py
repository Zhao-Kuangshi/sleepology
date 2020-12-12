# -*- coding: utf-8 -*-


class ModeError(Exception):
    '''
    When training, both `train_set` and `test_set` are required. And these
    dataset may be elaborately balanced and randomly shuffled. So please use
    `Sample.train_set()` and `Sample.test_set()`.

    However, when predicting, there are no `train_set` or `test_set`. And it is
    impossible to balance data or shuffle data. Please use `Sample.sample()` to
    just sample data without any other process.
    '''
    pass


class LackOfParameterError(Exception):
    '''
    Raised when sampling data. Different sample `unit` and sample `mode` needs
    different parameters. Please make sure your input meets the need of 
    `Sample` object.
    '''
    pass


class LackOfLabelDictError(Exception):
    '''
    Raised when you input dataset to a `Sample` instance and `y` element in
    that dataset does not have a label dict. Please use `Dataset.one_hot()`
    function to create one or manually set one.
    '''
    pass


class DataStateError(Exception):
    '''
    Raised when data in the dataset is not in a correct state, such as `ERROR`
    state. Please make sure you have tried effective `check` process and only
    input proper data into pipeline.
    '''
    pass


class BrokenTimestepError(Exception):
    '''
    Raised when sampling data in epochs. When epoch at edge cannot fill the
    required timestep to the full, and `epoch_padding == False`, the sample
    function will raise `BrokenTimestepError`, and the best way to handle this
    exception is using `continue` statement and going into next iteration to
    skip (also dispose) this epoch.
    '''
    pass


class KFoldError(Exception):
    '''
    Raised when iteration is not correctly set.
    '''
    pass


class TaskError(Exception):
    '''
    Raised when operation ordered in an improper task.
    '''
    pass