# 2019年3月27日使用，对睡眠脑多导进行分期。
# 网络结构按照cldnn0902实验二的网络结构

import tensorflow as tf;
import json;
import os;
import dataset;
import h5py;

class History(object):
    def __init__(self, save_path, name):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path
        self.name = name
        self.history = {}
    
    def add(self, fold, history):
        self.history[fold] = history
    
    def save(self):
        pos = os.path.join(self.save_path, self.name + '.history')
        disk_file = h5py.File(pos, 'w')
        for fold in self.history.keys():
            disk_file.create_group(fold)
            for attr in self.history[fold].keys():
                disk_file[fold].create_dataset(attr, 
                         data = self.history[fold][attr])

class Train(object):
    def __init__(self, save_path, name = 'train'):
        self.history = {};
        if not os.path.exists(save_path):
            os.mkdir(save_path);
        self.save_path = save_path;
        self.name = name;
        
    def set_training_param(self, batch_size, learning_rate, training_epoch,
                     epoch_steps, test_steps):
        ## 设置一个batch的大小，请考虑自己计算机的实际性能和数据集的大小
        self.batch_size = batch_size;
        self.learning_rate = learning_rate;
        self.training_epoch = training_epoch;
        self.epoch_steps = epoch_steps;
        self.test_steps = test_steps; # 规定最后测试多少个test batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size;
        
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate;
        
    def set_training_epoch(self, training_epoch):
        self.training_epoch = training_epoch;
    
    def set_epoch_steps(self, epoch_steps):
        self.epoch_steps = epoch_steps;
    
    def set_test_steps(self, test_steps):
        self.test_steps = test_steps;
        
    def __check_shape(self):
        # 2020-3-17 在训练开始前确认所有要素都齐了
        assert self.model.input_shape[1:] == self.data_input_shape;
        assert self.model.output_shape[1:] == self.data_output_shape;
        
    def feed(self, data, n_splits, data_name = None, tmin = 0, tmax = 0, padding = False):
        self.n_splits = n_splits;
        self.feed = data.k_fold(n_splits, data_name, tmin, tmax, padding);
        self.data_input_shape = data.x_input_shape;
        self.data_output_shape = data.y_output_shape;
    
    def set_model(self, model):
        self.model = model;

    def __cross_validation(self):
        pass

    def save(self):
        pos = os.path.join(self.save_path, self.name + '.history');
        disk_file = h5py.File(pos, 'w');
        for fold in self.history.keys():
            disk_file.create_group(fold);
            for attr in self.history[fold].keys():
                disk_file[fold].create_dataset(attr, 
                         data = self.history[fold][attr]);

    def train(self, cross_validation = True): 
        fold_index = 0;
        
        for train_dataset, test_dataset in self.feed.take(self.n_splits if cross_validation else 1):
        # train_dataset是训练集，test_dataset是测试集。train_dataset[0]和test_dataset[0]是data，四维数组；train[1]和test[1]是label，二维数组。
            
            fold_index += 1;
            
            checkpoint_file = os.path.join(self.save_path, 'cp_' + '{:02d}'.format(fold_index) + '{epoch:03d}_{val_loss:.4f}.h5');
            
            train_dataset = tf.data.Dataset.from_tensors(train_dataset).repeat().shuffle(1024).apply(tf.data.experimental.unbatch()).batch(self.batch_size);
            test_dataset = tf.data.Dataset.from_tensors(test_dataset).repeat().shuffle(1024).apply(tf.data.experimental.unbatch()).batch(self.batch_size);
            
            print('lr = ' + str(self.learning_rate) + ';');
            print('batch_size = ' + str(self.batch_size) + ';');
            
            ckpt = tf.keras.callbacks.ModelCheckpoint(checkpoint_file,verbose=1, save_weights_only=True, period=1);
            
            self.model.compile(optimizer = tf.keras.optimizers.Adam(lr = self.learning_rate), loss= 'categorical_crossentropy', metrics = ['accuracy']);
            
            # model.load_weights(os.path.join(save_path, 'test_weights.h5'));
            hist = self.model.fit(train_dataset, epochs = self.training_epoch, callbacks=[ckpt, tf.keras.callbacks.TensorBoard(log_dir = self.save_path)], steps_per_epoch = self.epoch_steps, validation_data = test_dataset, validation_steps = self.test_steps);
            self.history[str(fold_index)] = dict(hist.history);
            
            # Save weights to a HDF5 file
            self.model.save(self.save_path);
            
class NeuralNetworkTrain(Train):
    def __init__(self, save_path, name = 'train'):
        self.history = {};
        if not os.path.exists(save_path):
            os.mkdir(save_path);
        self.save_path = save_path;
        self.name = name;
        
    def set_training_param(self, batch_size, learning_rate, training_epoch,
                     epoch_steps, test_steps):
        ## 设置一个batch的大小，请考虑自己计算机的实际性能和数据集的大小
        self.batch_size = batch_size;
        self.learning_rate = learning_rate;
        self.training_epoch = training_epoch;
        self.epoch_steps = epoch_steps;
        self.test_steps = test_steps; # 规定最后测试多少个test batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size;
        
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate;
        
    def set_training_epoch(self, training_epoch):
        self.training_epoch = training_epoch;
    
    def set_epoch_steps(self, epoch_steps):
        self.epoch_steps = epoch_steps;
    
    def set_test_steps(self, test_steps):
        self.test_steps = test_steps;
        
    def __check_shape(self):
        # 2020-3-17 在训练开始前确认所有要素都齐了
        assert self.model.input_shape[1:] == self.data_input_shape;
        assert self.model.output_shape[1:] == self.data_output_shape;
        
    def feed(self, data, n_splits, data_name = None, tmin = 0, tmax = 0, padding = False):
        self.n_splits = n_splits;
        self.feed = data.k_fold(n_splits, data_name, tmin, tmax, padding);
        self.data_input_shape = data.x_input_shape;
        self.data_output_shape = data.y_output_shape;
    
    def set_model(self, model):
        self.model = model;
                
    def save(self):
        pos = os.path.join(self.save_path, self.name + '.history');
        disk_file = h5py.File(pos, 'w');
        for fold in self.history.keys():
            disk_file.create_group(fold);
            for attr in self.history[fold].keys():
                disk_file[fold].create_dataset(attr, 
                         data = self.history[fold][attr]);

    def train(self, cross_validation = True): 
        fold_index = 0;
        
        for train_dataset, test_dataset in self.feed.take(self.n_splits if cross_validation else 1):
        # train_dataset是训练集，test_dataset是测试集。train_dataset[0]和test_dataset[0]是data，四维数组；train[1]和test[1]是label，二维数组。
            
            fold_index += 1;
            
            checkpoint_file = os.path.join(self.save_path, 'cp_' + '{:02d}'.format(fold_index) + '{epoch:03d}_{val_loss:.4f}.h5');
            
            train_dataset = tf.data.Dataset.from_tensors(train_dataset).repeat().shuffle(1024).apply(tf.data.experimental.unbatch()).batch(self.batch_size);
            test_dataset = tf.data.Dataset.from_tensors(test_dataset).repeat().shuffle(1024).apply(tf.data.experimental.unbatch()).batch(self.batch_size);
            
            print('lr = ' + str(self.learning_rate) + ';');
            print('batch_size = ' + str(self.batch_size) + ';');
            
            ckpt = tf.keras.callbacks.ModelCheckpoint(checkpoint_file,verbose=1, save_weights_only=True, period=1);
            
            self.model.compile(optimizer = tf.keras.optimizers.Adam(lr = self.learning_rate), loss= 'categorical_crossentropy', metrics = ['accuracy']);
            
            # model.load_weights(os.path.join(save_path, 'test_weights.h5'));
            hist = self.model.fit(train_dataset, epochs = self.training_epoch, callbacks=[ckpt, tf.keras.callbacks.TensorBoard(log_dir = self.save_path)], steps_per_epoch = self.epoch_steps, validation_data = test_dataset, validation_steps = self.test_steps);
            self.history[str(fold_index)] = dict(hist.history);
            
            # Save weights to a HDF5 file
            self.model.save(self.save_path);