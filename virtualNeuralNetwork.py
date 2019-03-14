#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
from .cuda_utils import GPU_count

class VirtualNN(object):
    def __init__(self,param):
        self.param = param
        self.graph = tf.Graph()
        self.build_model()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
#        config.gpu_options.per_process_gpu_memory_fraction = 0.22
        gpu_count = GPU_count()
        if gpu_count > 0:
            import os
            gpu_num = os.getpid() % gpu_count
            config.gpu_options.visible_device_list= str(gpu_num)
        self.sess = tf.Session(config=config,graph=self.graph)
        with self.graph.as_default():
            self.saver = tf.train.Saver()

    def __del__(self):
        self.sess.close()
        del self.sess
        del self.param
        del self.graph

    def test(self,test_data,test_label,verbal_flag=False):
        with self.graph.as_default():
            sess = self.sess
            test_acc_sum = 0
            for data,label in zip(test_data,test_label):
                test_acc = sess.run(self.accuracy,feed_dict={self.input_data:np.stack([data],axis=0),self.input_label:np.stack([label],axis=0),self.train_flag:False})
                test_acc_sum += test_acc * data.shape[0]
            test_acc = test_acc_sum / np.sum([data.shape[0] for data in test_data])
            if verbal_flag:
                print('final test accuracy: %f'%test_acc)
        return test_acc

    def train(self,train_data,train_label,test_data=None,test_label=None,verbal_flag=False):
        if verbal_flag:
            print(self.param)
        test_flag = not ((test_data is None) or (test_label is None))
        with self.graph.as_default():
            sess = self.sess
            param = self.param
            train_trail_num = len(train_data)
            if test_flag:
                test_trail_num = len(test_data)

            sess.run(tf.global_variables_initializer())
            max_train_acc = 0
            min_train_loss = 1e7
            wait_epoch = 0

            for en in range(1,param.epoch_num+1):
                sample_index = np.random.choice(train_trail_num,param.batch_size)
                temporal_len = min(param.temporal_len, min([train_data[index].shape[0] for index in sample_index]))-1
                train_data_samples = np.zeros([param.batch_size,temporal_len,param.input_dim])
                train_label_samples = np.zeros([param.batch_size,temporal_len,param.label_dim])
                for i,index in enumerate(sample_index):
                    start_index = np.random.choice(train_data[index].shape[0]-temporal_len)
                    train_data_samples[i] = train_data[index][start_index:start_index+temporal_len]
                    train_label_samples[i] = train_label[index][start_index:start_index+temporal_len]

                _,train_loss,train_acc = sess.run([self.adam,self.loss,self.accuracy],feed_dict={self.input_data:train_data_samples,self.input_label:train_label_samples,self.train_flag:True})
                if verbal_flag and en%50 == 0:
                    if test_flag:
                        sample_index = np.random.choice(test_trail_num,param.batch_size)
                        temporal_len = min(param.temporal_len, min([test_data[index].shape[0] for index in sample_index])-1)
                        test_data_samples = np.zeros([param.batch_size,temporal_len,param.input_dim])
                        test_label_samples = np.zeros([param.batch_size,temporal_len,param.label_dim])
                        for i,index in enumerate(sample_index):
                            start_index = np.random.choice(test_data[index].shape[0]-temporal_len)
                            test_data_samples[i] = test_data[index][start_index:start_index+temporal_len]
                            test_label_samples[i] = test_label[index][start_index:start_index+temporal_len]
                        test_loss,test_acc = sess.run([self.loss,self.accuracy],feed_dict={self.input_data:test_data_samples,self.input_label:test_label_samples,self.train_flag:False})
                        print(str(en)+'/'+str(param.epoch_num)+': ',train_loss,train_acc,test_loss,test_acc)
                    else:
                        print(str(en)+'/'+str(param.epoch_num)+': ',train_loss,train_acc)
                if train_loss + param.min_delta < min_train_loss:
                    min_train_loss = train_loss
                    wait_epoch = 0
                else:
                    if wait_epoch >= param.patience:
                        break;
                    else:
                        wait_epoch += 1
    #                if train_acc - param.min_delta > max_train_acc:
    #                    max_train_acc = train_acc
    #                    wait_epoch = 0
    #                else:
    #                    if wait_epoch >= param.patience:
    #                        break;
    #                    else:
    #                        wait_epoch += 1


    def train_test(self,train_data,train_label,test_data,test_label,verbal_flag=False):
        self.train(train_data,train_label,test_data,test_label,verbal_flag)
        return self.test(test_data,test_label,verbal_flag)

    def cross_session_val_with_fs(self,data,labels,select_func,verbal_flag=False):
        session_num = len(data)
        accuracies = [0]*session_num
        for session_index in range(session_num):
            train_data_sessions = data[:session_index] + data[session_index+1:]
            train_label_sessions = labels[:session_index] + labels[session_index+1:]
            train_data = []
            for sess in train_data_sessions:
                train_data += sess
            train_label = []
            for sess in train_label_sessions:
                train_label += sess
            test_data = data[session_index]
            test_label = labels[session_index]

            select_index = select_func(np.concatenate(train_data,axis=0),np.concatenate(train_label,axis=0),session_index)
            train_data = [data[:,select_index] for data in train_data]
            test_data = [data[:,select_index] for data in test_data]

            data_mean = np.mean(np.concatenate(train_data,axis=0),axis=0)
            data_std = np.std(np.concatenate(train_data,axis=0),axis=0)
            train_data = [(data - data_mean) / data_std for data in train_data]
            test_data = [(data - data_mean) / data_std for data in test_data]

            accuracies[session_index] = self.train_test(train_data,train_label,test_data,test_label,verbal_flag)
        return accuracies

    def cross_session_val(self,data,labels,verbal_flag=False):
        session_num = len(data)
        accuracies = [0]*session_num
        for session_index in range(session_num):
            train_data_sessions = data[:session_index] + data[session_index+1:]
            train_label_sessions = labels[:session_index] + labels[session_index+1:]
            train_data = []
            for sess in train_data_sessions:
                train_data += sess
            train_label = []
            for sess in train_label_sessions:
                train_label += sess
            test_data = data[session_index]
            test_label = labels[session_index]

            data_mean = np.mean(np.concatenate(train_data,axis=0),axis=0)
            data_std = np.std(np.concatenate(train_data,axis=0),axis=0)
            train_data = [(data - data_mean) / data_std for data in train_data]
            test_data = [(data - data_mean) / data_std for data in test_data]

            accuracies[session_index] = self.train_test(train_data,train_label,test_data,test_label,verbal_flag)
        return accuracies

    def cross_session_train_test(self,train_data,train_labels,test_data,test_labels,verbal_flag=False):
        session_num = len(train_data)
        accuracies = [0]*session_num

        # standardization
        train_data_unnorm_sum = [d for data in train_data for d in data]
        data_mean = np.mean(np.concatenate(train_data_unnorm_sum,axis=0),axis=0)
        data_std = np.std(np.concatenate(train_data_unnorm_sum,axis=0),axis=0)
        train_data = [[(d - data_mean) / data_std for d in data] for data in train_data]
        test_data = [[(d - data_mean) / data_std for d in data] for data in test_data]

        # append session one-hot features
        train_data = [[np.concatenate([d,np.stack([np.eye(session_num)[index]]*d.shape[0],axis=0)],axis=1) for d in data] for index,data in enumerate(train_data)]
        test_data = [[np.concatenate([d,np.stack([np.eye(session_num)[index]]*d.shape[0],axis=0)],axis=1) for d in data] for index,data in enumerate(test_data)]

        # flatten lists
        train_data_sum = [d for data in train_data for d in data]
        train_labels_sum = [l for labels in train_labels for l in labels]
        if verbal_flag:
            test_data_sum = [d for data in test_data for d in data]
            test_labels_sum = [l for labels in test_labels for l in labels]
        else:
            test_data_sum = None
            test_labels_sum = None

        # train & test
        self.train(train_data_sum,train_labels_sum,test_data_sum,test_labels_sum,verbal_flag)
        for session_index in range(session_num):
            accuracies[session_index] = self.test(test_data[session_index],test_labels[session_index],verbal_flag)
        return accuracies

    def save_variables(self,model_path):
        self.saved_path = self.saver.save(self.sess,model_path)
        return self.saved_path

    def restore_variables(self,model_path=None):
        self.saver.restore(self.sess,self.saved_path if model_path is None else model_path)
