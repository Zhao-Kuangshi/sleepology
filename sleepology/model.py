# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:26:18 2020

@author: 赵匡是
"""
import os
import h5py
import numpy
import tensorflow as tf
from abc import ABC, abstractmethod
from packaging import version
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from train import History


class Model(object):
    @abstractmethod
    def train(self):
        pass
    
    def validate(self):
        pass
    
    def feed(self, data, n_splits, data_name = None, tmin = 0, tmax = 0, padding = False):
        self.n_splits = n_splits
        self.feed = data.k_fold(n_splits, data_name, tmin, tmax, padding, array_type = None)
        self.data_input_shape = data.x_input_shape
        self.data_output_shape = data.y_output_shape

class LDAModel(Model):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def train(self, save_path, name, cross_validation = True, reshape = None): 
        history = History(save_path, name)
        
        fold_index = 0
        
        for train_dataset, test_dataset in self.feed.take(self.n_splits if cross_validation else 1):
        # train_dataset是训练集，test_dataset是测试集。train_dataset[0]和test_dataset[0]是data，四维数组；train[1]和test[1]是label，二维数组。
            fold_index += 1
            lda_model = LinearDiscriminantAnalysis()
            
            train_data = numpy.array(train_dataset[0])
            train_label = numpy.array(train_dataset[1])
            test_data = numpy.array(test_dataset[0])
            test_label = numpy.array(test_dataset[1])
            
            lda_model.fit(train_data.reshape((train_data.shape[0], -1)), train_label)
            
            # model.load_weights(os.path.join(save_path, 'test_weights.h5'))
            r = {}
            r['acc'] = lda_model.score(train_data.reshape((train_data.shape[0], -1)), train_label)
            print('acc = ' + str(r['acc']))
            r['val_acc'] = lda_model.score(test_data.reshape((test_data.shape[0], -1)), test_label)
            print('val_acc = ' + str(r['val_acc']))
            history.add(str(fold_index), r)
            
            # Save weights to a HDF5 file
            # self.model.save(self.save_path)
        history.save()

class PCALDAModel(Model):
    def __init__(self, model_name, pca_components=None):
        self.model_name = model_name
        self.pca_components = pca_components
    
    def train(self, save_path, name, cross_validation = True, reshape = None): 
        history = History(save_path, name)
        
        fold_index = 0
        
        for train_dataset, test_dataset in self.feed.take(self.n_splits if cross_validation else 1):
        # train_dataset是训练集，test_dataset是测试集。train_dataset[0]和test_dataset[0]是data，四维数组；train[1]和test[1]是label，二维数组。
            fold_index += 1
            pca_model = PCA(n_components=self.pca_components)
            lda_model = LinearDiscriminantAnalysis()
            
            train_data = numpy.array(train_dataset[0])
            train_label = numpy.array(train_dataset[1])
            test_data = numpy.array(test_dataset[0])
            test_label = numpy.array(test_dataset[1])
            
            pca_model.fit(train_data.reshape((train_data.shape[0], -1)))
            train_pc = pca_model.transform(train_data.reshape((train_data.shape[0], -1)))
            lda_model.fit(train_pc, train_label)
            
            # model.load_weights(os.path.join(save_path, 'test_weights.h5'))
            r = {}
            r['acc'] = lda_model.score(train_pc, train_label)
            print('acc = ' + str(r['acc']))
            test_pc = pca_model.transform(test_data.reshape((test_data.shape[0], -1)))
            r['val_acc'] = lda_model.score(test_pc, test_label)
            print('val_acc = ' + str(r['val_acc']))
            history.add(str(fold_index), r)
            
            # Save weights to a HDF5 file
            # self.model.save(self.save_path)
        history.save()

class SVMModel(Model):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def train(self, save_path, name, cross_validation = True, reshape = None): 
        history = History(save_path, name)
        
        fold_index = 0
        
        for train_dataset, test_dataset in self.feed.take(self.n_splits if cross_validation else 1):
        # train_dataset是训练集，test_dataset是测试集。train_dataset[0]和test_dataset[0]是data，四维数组；train[1]和test[1]是label，二维数组。
            fold_index += 1
            svm_model = SVC()
            
            train_data = numpy.array(train_dataset[0])
            train_label = numpy.array(train_dataset[1])
            test_data = numpy.array(test_dataset[0])
            test_label = numpy.array(test_dataset[1])
            
            svm_model.fit(train_data.reshape((train_data.shape[0], -1)), train_label)
            
            # model.load_weights(os.path.join(save_path, 'test_weights.h5'))
            r = {}
            r['acc'] = svm_model.score(train_data.reshape((train_data.shape[0], -1)), train_label)
            print('acc = ' + str(r['acc']))
            r['val_acc'] = svm_model.score(test_data.reshape((train_data.shape[0], -1)), test_label)
            print('val_acc = ' + str(r['val_acc']))
            history.add(str(fold_index), r)
            
            # Save weights to a HDF5 file
            # self.model.save(self.save_path)
        history.save()

class NeuralNetworkModel(Model, tf.keras.Model):
    def __init__(self, model_name):
        super(NeuralNetworkModel, self).__init__()
        self.model_name = model_name
        
    
    def save(self, filepath, overwrite=True, include_optimizer=True, 
             save_format='h5', signatures=None, options=None):
        filepath = os.path.join(filepath, self.model_name + '.model')
        if version.parse(tf.__version__) < version.parse('2.0.0'):
            super(SequentialNeuralNetworkModel, self).save(filepath, overwrite, 
             include_optimizer)
        else:
            super(SequentialNeuralNetworkModel, self).save(filepath, overwrite, 
             include_optimizer, save_format, signatures, options)
    
    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        
        model = tf.keras.models.load_model(filepath, 
                                           custom_objects, compile)
        model.__class__ = NeuralNetworkModel
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        model.model_name = model_name
        return model
    
    def set_training_param(self, batch_size, learning_rate, training_epoch,
                     epoch_steps, test_steps):
        ## 设置一个batch的大小，请考虑自己计算机的实际性能和数据集的大小
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_epoch = training_epoch
        self.epoch_steps = epoch_steps
        self.test_steps = test_steps # 规定最后测试多少个test batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        
    def set_training_epoch(self, training_epoch):
        self.training_epoch = training_epoch
    
    def set_epoch_steps(self, epoch_steps):
        self.epoch_steps = epoch_steps
    
    def set_test_steps(self, test_steps):
        self.test_steps = test_steps
    
    def feed(self, data, n_splits, data_name = None, tmin = 0, tmax = 0, padding = False):
        self.n_splits = n_splits
        self.feed = data.k_fold(n_splits, data_name, tmin, tmax, padding)
        self.data_input_shape = data.x_input_shape
        self.data_output_shape = data.y_output_shape
        
    
    def train(self, save_path, name, cross_validation = True): 
        history = History(save_path, name)
        
        fold_index = 0
        if version.parse(tf.__version__) < version.parse('1.13'):
            self.compile(optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate), loss= 'categorical_crossentropy', metrics = ['accuracy'])
        else:
            self.compile(optimizer = tf.keras.optimizers.Adam(lr = self.learning_rate), loss= 'categorical_crossentropy', metrics = ['accuracy'])
        
        for train_dataset, test_dataset in self.feed.take(self.n_splits if cross_validation else 1):
        # train_dataset是训练集，test_dataset是测试集。train_dataset[0]和test_dataset[0]是data，四维数组；train[1]和test[1]是label，二维数组。
            
            fold_index += 1
            
            checkpoint_file = os.path.join(save_path, 'cp_' + '{:02d}'.format(fold_index) + '{epoch:03d}_{val_loss:.4f}.h5')
            
            train_dataset = tf.data.Dataset.from_tensors(train_dataset).repeat().shuffle(1024).apply(tf.data.experimental.unbatch()).batch(self.batch_size)
            test_dataset = tf.data.Dataset.from_tensors(test_dataset).repeat().shuffle(1024).apply(tf.data.experimental.unbatch()).batch(self.batch_size)
            
            print('lr = ' + str(self.learning_rate) + '')
            print('batch_size = ' + str(self.batch_size) + '')
            
            ckpt = tf.keras.callbacks.ModelCheckpoint(checkpoint_file,verbose=1, save_weights_only=True, period=1)
            
            # model.load_weights(os.path.join(save_path, 'test_weights.h5'))
            hist = self.fit(train_dataset, epochs = self.training_epoch, callbacks=[ckpt, tf.keras.callbacks.TensorBoard(log_dir = save_path)], steps_per_epoch = self.epoch_steps, validation_data = test_dataset, validation_steps = self.test_steps)
            history.add(str(fold_index), dict(hist.history))
            
            # Save weights to a HDF5 file
            self.save(save_path)
        
        history.save()
        
class SequentialNeuralNetworkModel(tf.keras.Sequential):
    def __init__(self, model_name):
        super(SequentialNeuralNetworkModel, self).__init__()
        self.model_name = model_name
    
    def save(self, filepath, overwrite=True, include_optimizer=True, 
             save_format='h5', signatures=None, options=None):
        filepath = os.path.join(filepath, self.model_name + '.model')
        if version.parse(tf.__version__) < version.parse('2.0.0'):
            super(SequentialNeuralNetworkModel, self).save(filepath, overwrite, 
             include_optimizer)
        else:
            super(SequentialNeuralNetworkModel, self).save(filepath, overwrite, 
             include_optimizer, save_format, signatures, options)
    
    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        
        model = tf.keras.models.load_model(filepath, 
                                           custom_objects, compile)
        model.__class__ = SequentialNeuralNetworkModel
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        model.model_name = model_name
        return model
    
    def set_training_param(self, batch_size, learning_rate, training_epoch,
                     epoch_steps, test_steps):
        ## 设置一个batch的大小，请考虑自己计算机的实际性能和数据集的大小
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_epoch = training_epoch
        self.epoch_steps = epoch_steps
        self.test_steps = test_steps # 规定最后测试多少个test batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        
    def set_training_epoch(self, training_epoch):
        self.training_epoch = training_epoch
    
    def set_epoch_steps(self, epoch_steps):
        self.epoch_steps = epoch_steps
    
    def set_test_steps(self, test_steps):
        self.test_steps = test_steps
    
    def feed(self, data, n_splits, data_name = None, tmin = 0, tmax = 0, padding = False):
        self.n_splits = n_splits
        self.feed = data.k_fold(n_splits, data_name, tmin, tmax, padding)
        self.data_input_shape = data.x_input_shape
        self.data_output_shape = data.y_output_shape
        
    
    def train(self, save_path, name, cross_validation = True): 
        history = History(save_path, name)
        
        fold_index = 0
        if version.parse(tf.__version__) < version.parse('1.13'):
            self.compile(optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate), loss= 'categorical_crossentropy', metrics = ['accuracy'])
        else:
            self.compile(optimizer = tf.keras.optimizers.Adam(lr = self.learning_rate), loss= 'categorical_crossentropy', metrics = ['accuracy'])
        
        for train_dataset, test_dataset in self.feed.take(self.n_splits if cross_validation else 1):
        # train_dataset是训练集，test_dataset是测试集。train_dataset[0]和test_dataset[0]是data，四维数组；train[1]和test[1]是label，二维数组。
            
            fold_index += 1
            
            checkpoint_file = os.path.join(save_path, 'cp_' + '{:02d}'.format(fold_index) + '{epoch:03d}_{val_loss:.4f}.h5')
            
            train_dataset = tf.data.Dataset.from_tensors(train_dataset).repeat().shuffle(1024).apply(tf.data.experimental.unbatch()).batch(self.batch_size)
            test_dataset = tf.data.Dataset.from_tensors(test_dataset).repeat().shuffle(1024).apply(tf.data.experimental.unbatch()).batch(self.batch_size)
            
            print('lr = ' + str(self.learning_rate) + '')
            print('batch_size = ' + str(self.batch_size) + '')
            
            ckpt = tf.keras.callbacks.ModelCheckpoint(checkpoint_file,verbose=1, save_weights_only=True, period=1)
            
            # model.load_weights(os.path.join(save_path, 'test_weights.h5'))
            hist = self.fit(train_dataset, epochs = self.training_epoch, callbacks=[ckpt, tf.keras.callbacks.TensorBoard(log_dir = save_path)], steps_per_epoch = self.epoch_steps, validation_data = test_dataset, validation_steps = self.test_steps)
            history.add(str(fold_index), dict(hist.history))
            
            # Save weights to a HDF5 file
            self.save(save_path)
        
        history.save()
