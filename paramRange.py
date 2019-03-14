#!/usr/bin/env python
# coding=utf-8

import numpy as np

class ParamItem:
    def __init__(self,name,default,float_flag=True):
        '''
        name is the parameter's formal name, of type string
        default is the default value, of type int or float
        float_flag is whether the paramter is float or int
        '''
        self.name = name
        self.default = default
        self.float_flag = float_flag
        self._assign(default)
    def _assign(self,value):
        if self.float_flag:
            self.value = value*1.
        else:
            self.value = int(value)
    def assign_value(self,value=None):
        if value is None:
            self._assign(self.default)
        else:
            self._assign(value)
    def __str__(self):
        return self.name+': '+str(self.value)
    def __repr__(self):
        return self.__str__()

class VirtualParam(object):
    '''
    Virtual base class for Param classes of each model.
    This class is a virtual class, created to save duplicated codes in different models, and can not be used directly.
    The upper class, which should be written in the model file, should rewrite the __init__ function.
        In that function, self.paramters is created and reassigned, which should be one dictionary,
        whose key is the paramter name and value is of type parameter, which contains the parameter value, parameter full name and parameter float flag.
        Also, in the __init__ function, self.model_name should be reassigned, which contains the full model name
    '''
    def __init__(self):
        self.parameters = {}
        self.model_name = 'no model name assigned'
    def assign_value_per_parameter(self,param_name,value):
        self.parameters[param_name].value = value
    def assign_value(self,param_list=None):
        if param_list is None:
            param_list = [None]*len(self.parameters)
        keys = list(self.parameters.keys())
        keys.sort()
        for i,key in enumerate(keys):
            self.parameters[key].assign_value(param_list[i])
    def __setattr__(self,key,value):
        if key == 'parameters' or key == 'model_name':
            object.__setattr__(self,key,value)
        elif key in self.parameters:
            self.parameters[key].assign_value(value)
        else:
            raise AttributeError
    def __getattribute__(self,key):
        if key in object.__getattribute__(self,'parameters').keys():
            return self.parameters[key].value
        else:
            return object.__getattribute__(self,key)
    def __str__(self):
        pstr = '-----------------------------------------------------------------------------------'+'\r\n'
        pstr += self.model_name+' paramters: '+'\r\n'
        keys = list(self.parameters.keys())
        keys.sort()
        for key in keys:
            pstr += str(self.parameters[key])+'\r\n'
        pstr += '-----------------------------------------------------------------------------------'+'\r\n'
        return pstr
    def __repr__(self):
        return self.__str__()

class ParamRangeItem:
    def __init__(self,low,high,log_flag=False):
        self.low = low
        self.high = high
        self.log_flag = log_flag
    def __del__(self):
        del self.low
        del self.high
        del self.log_flag
    def sample(self,size):
        result = np.random.random_sample(size)*(self.high-self.low)+self.low
        if self.log_flag:
            return 10**result
        else:
            return result
    def assign_value(self,low,high,log_flag=False):
        self.low = low
        self.high = high
        self.log_flag = log_flag

class VirtualParamRange(object):
    def __init__(self):
        self.param_ranges = {}
    def sample(self,size):
        keys = list(self.param_ranges.keys())
        keys.sort()
        return np.stack([self.param_ranges[key].sample(size) for key in keys],-1)
    def assign_value_per_parameter(self,param_name,low,high,log_flag=False):
        self.param_ranges[param_name].assign_value(low,high,log_flag)

