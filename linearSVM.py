import numpy as np
from . import paramRange as _paramRange
import os,sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.join(curdir,'liblinear'),'python'))
from liblinearutil import *

class _Param(_paramRangeVirtualParam):
    def __init__(self):
        self.model_name = 'LinearSVC'
        self.parameters = {
                'c':_paramRangeParamItem('c',1.0)
                }

class _ParamRange(_paramRangeVirtualParamRange):
    def __init__(self):
        self.param_ranges = {
                'c':_paramRangeParamRangeItem(-7,3,True)
                }

class Model:
    def __init__(self,param):
        self.param = param
        self.build_model()
    def build_model(self):
        pass
    def cross_session_val(self,data,labels):
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

            # sample_index = np.random.choice(train_data.shape[0],10000)
            # train_data = train_data[sample_index]
            # train_label = train_label[sample_index]

            data_mean = np.mean(train_data,axis=0)
            data_std = np.std(train_data,axis=0)
            train_data = (train_data - data_mean) / data_std
            test_data = (test_data - data_mean) / data_std

            prob = problem(train_label.tolist(),train_data.tolist())
            model =train(prob,parameter('-c %f'%self.param.c))
            p_label, p_acc, p_val = predict(test_label.tolist(),test_data.tolist(),model)
            accuracies[session_index] = p_acc[0]
        return accuracies
