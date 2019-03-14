from . import paramRange as _paramRange
from . import virtualNeuralNetwork as _virtualNeuralNetwork
import tensorflow as _tf
from . import nn_parts as _nn

class _Param(_paramRange.VirtualParam):
    def __init__(self):
        self.model_name = 'gcnn+drop+FC'
        ParamItem = _paramRange.ParamItem
        self.parameters = {
                'input_dim':ParamItem('input_dimension',40,False),
                'input_fs':ParamItem('input_filter_size',1,False),
                'label_dim':ParamItem('label_dimension',2,False),
                'temporal_len':ParamItem('temporal_length',30,False),
                'ks':ParamItem('kernel_size',5,False),
                'fs':ParamItem('hidden_size',32,False),
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
                'input_fs':ParamRangeItem(1,1,False),
                'label_dim':ParamRangeItem(2,2,False),
                'temporal_len':ParamRangeItem(10,200,False),
                'ks':ParamRangeItem(1,40,False),
                'fs':ParamRangeItem(4,128,False),
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
        node_num = int(param.input_dim/param.input_fs)
        with self.graph.as_default():
            with _tf.name_scope('Input'):
                self.input_data = _tf.placeholder(_tf.float32,[None,None,param.input_dim],name='input_data')
                self.input_label = _tf.placeholder(_tf.float32,[None,None,param.label_dim],name='label')
                self.train_flag = _tf.placeholder(_tf.bool,name='train_flag')

            with _tf.name_scope('GCNN'):
                self.reshaped_input_data = _tf.reshape(self.input_data,[-1,node_num,param.input_fs])
                self.gcnn_layer = _nn.gnn.FastGCNN(param.ks,param.input_fs,param.fs,node_num,name_scope='FastGCNNLayer')
                self.gcnn_out = self.gcnn_layer(self.reshaped_input_data)
                print(self.gcnn_out.shape)

            with _tf.name_scope('Dropout'):
                self.gcnn_out_drop = _tf.layers.dropout(self.gcnn_out,param.dp,training=self.train_flag,name='gcnn_out_drop')
                print(self.gcnn_out_drop.shape)

            with _tf.name_scope('Flat'):
                self.gcnn_out_drop_flat = _tf.reshape(self.gcnn_out_drop,[-1,node_num*param.fs],name='gcnn_out_drop_flat')
                self.input_label_flat= _tf.reshape(self.input_label,[-1,param.label_dim],name='input_label_flat')

            with _tf.name_scope('Dense'):
                self.dense_layer = _nn.fc.Dense(node_num*param.fs,param.label_dim,activation_func=_linear_activation,name_scope='DenseLayer')
                self.logits = self.dense_layer(self.gcnn_out_drop_flat)
                print(self.logits.shape)

            with _tf.name_scope('Loss'):
                self.cross_entropy_loss = _tf.reduce_mean(_tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label_flat,logits=self.logits,dim=-1),name='softmax_loss')
                self.l2_loss = _tf.reduce_sum([self.dense_layer.get_l2_loss()],name='l2_loss')
                self.loss = _tf.add(self.cross_entropy_loss,param.l2*self.l2_loss,name='loss')
                self.adam = _tf.train.AdamOptimizer(param.lr).minimize(self.loss)

            with _tf.name_scope('Accuracy'):
                self.accuracy = _tf.reduce_mean(_tf.cast(_tf.equal(_tf.argmax(self.input_label_flat,axis=1),_tf.argmax(self.logits,axis=1)),_tf.float32),name='accuracy')
