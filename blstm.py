from . import paramRange as _paramRange
from . import virtualNeuralNetwork as _virtualNeuralNetwork
import tensorflow as _tf
from . import nn_parts as _nn

class _Param(_paramRange.VirtualParam):
    def __init__(self):
        self.model_name = 'BLSTM+drop+FC'
        ParamItem = _paramRange.ParamItem
        self.parameters = {
                'input_dim':ParamItem('input_dimension',40,False),
                'input_dim1':ParamItem('input_dimension1',40,False),
                'label_dim':ParamItem('label_dimension',2,False),
                'temporal_len':ParamItem('temporal_length',30,False),
                'hs1':ParamItem('hidden_size1',128,False),
                'hs2':ParamItem('hidden_size2',128,False),
                'dp':ParamItem('dropout_rate',0.5,True),
                'l2':ParamItem('l2_regularization_strength',1e-3,True),
                'lr':ParamItem('learning_rate',1e-3,True),
                'batch_size':ParamItem('batch_size',8,False),
                'epoch_num':ParamItem('epoch_number',100000,False),
                'min_delta':ParamItem('min_delta',1e-3,True),
                'patience':ParamItem('patience',120,False),
                }

class _ParamRange(_paramRange.VirtualParamRange):
    def __init__(self):
        ParamRangeItem = _paramRange.ParamRangeItem
        self.param_ranges = {
                'input_dim':ParamRangeItem(40,40,False),
                'input_dim1':ParamRangeItem(40,40,False),
                'label_dim':ParamRangeItem(2,2,False),
                'temporal_len':ParamRangeItem(10,200,False),
                'hs1':ParamRangeItem(16,1024,False),
                'hs2':ParamRangeItem(16,1024,False),
                'dp':ParamRangeItem(0.2,0.8,False),
                'l2':ParamRangeItem(-7,0,True),
                'lr':ParamRangeItem(-7,0,True),
                'batch_size':ParamRangeItem(16,16,False),
                'epoch_num':ParamRangeItem(100000,100000,False),
                'min_delta':ParamRangeItem(1e-3,1e-3,False),
                'patience':ParamRangeItem(3000,3000,False),
                }

def _linear_activation(x):
    return x

class Model(_virtualNeuralNetwork.VirtualNN):
    def build_model(self):
        param = self.param
        with self.graph.as_default():
            with _tf.name_scope('Input'):
                self.input_data = _tf.placeholder(_tf.float32,[None,None,param.input_dim],name='input_data')
                self.input_label = _tf.placeholder(_tf.float32,[None,None,param.label_dim],name='label')
                self.train_flag = _tf.placeholder(_tf.bool,name='train_flag')

            with _tf.name_scope('LSTM'):
                self.lstm_layer1 = _nn.rnn.LSTMCell(param.input_dim1,param.hs1,name_scope='LSTMLayer1')
                self.lstm_out1 = self.lstm_layer1(self.input_data[:,:,:param.input_dim1])
                self.lstm_layer2 = _nn.rnn.LSTMCell(param.input_dim-param.input_dim1,param.hs2,name_scope='LSTMLayer2')
                self.lstm_out2 = self.lstm_layer2(self.input_data[:,:,param.input_dim1:])
                self.lstm_out = _tf.concat([self.lstm_out1,self.lstm_out2],axis=-1,name='lstm_out')

            with _tf.name_scope('Dropout'):
                self.lstm_out_drop = _tf.layers.dropout(self.lstm_out,param.dp,training=self.train_flag,name='lstm_out_drop')

            with _tf.name_scope('Flat'):
                self.lstm_out_drop_flat = _tf.reshape(self.lstm_out_drop,[-1,param.hs1+param.hs2],name='lstm_out_drop_flat')
                self.input_label_flat= _tf.reshape(self.input_label,[-1,param.label_dim],name='input_label_flat')

            with _tf.name_scope('Dense'):
                self.dense_layer = _nn.fc.Dense(param.hs1+param.hs2,param.label_dim,activation_func=_linear_activation,name_scope='DenseLayer')
                self.logits = self.dense_layer(self.lstm_out_drop_flat)

            with _tf.name_scope('Loss'):
                self.cross_entropy_loss = _tf.reduce_mean(_tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label_flat,logits=self.logits,dim=-1),name='softmax_loss')
                self.l2_loss = _tf.reduce_sum([self.dense_layer.get_l2_loss()],name='l2_loss')
                self.loss = _tf.add(self.cross_entropy_loss,param.l2*self.l2_loss,name='loss')
                self.adam = _tf.train.AdamOptimizer(param.lr).minimize(self.loss)

            with _tf.name_scope('Accuracy'):
                self.accuracy = _tf.reduce_mean(_tf.cast(_tf.equal(_tf.argmax(self.input_label_flat,axis=1),_tf.argmax(self.logits,axis=1)),_tf.float32),name='accuracy')

