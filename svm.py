import numpy as np
from sklearn.svm import LinearSVC
from . import paramRange as _paramRange

class _Param(_paramRange.VirtualParam):
    def __init__(self):
        self.model_name = 'LinearSVC'
        self.parameters = {
                'c':_paramRange.ParamItem('c',1.0)
                }

class _ParamRange(_paramRange.VirtualParamRange):
    def __init__(self):
        self.param_ranges = {
                'c':_paramRange.ParamRangeItem(-7,3,True)
                }

class Model:
    def __init__(self,param):
        self.param = param
        self.build_model()
    def build_model(self):
        self.clf = LinearSVC(C=self.param.c)
    def cross_session_val(self,data,labels,verbal_flag=None):
        session_num = len(data)
        accuracies = [0]*session_num
        for session_index in range(session_num):
            train_data_sessions = data[:session_index] + data[session_index+1:]
            train_label_sessions = labels[:session_index] + labels[session_index+1:]
            test_data_session = data[session_index]
            test_label_session = labels[session_index]

            train_data = np.concatenate([np.concatenate(sess,axis=0) for sess in train_data_sessions],axis=0)
            train_label = np.concatenate([np.concatenate(sess,axis=0) for sess in train_label_sessions],axis=0)[:,1]
            test_data = np.concatenate(test_data_session,axis=0)
            test_label = np.concatenate(test_label_session,axis=0)[:,1]

            data_mean = np.mean(train_data,axis=0)
            data_std = np.std(train_data,axis=0)
            train_data = (train_data - data_mean) / data_std
            test_data = (test_data - data_mean) / data_std

            # print(train_data.shape,train_label.shape,test_data.shape,test_label.shape,data_mean.shape,data_std.shape)
            self.clf.fit(train_data,train_label)
            accuracies[session_index] = self.clf.score(test_data,test_label)
            print('Final accuracy: ',accuracies[session_index])
        return accuracies

