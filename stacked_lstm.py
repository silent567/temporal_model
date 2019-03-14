from . import paramRange as _paramRange
from . import virtualNeuralNetwork as _virtualNeuralNetwork
import tensorflow as _tf
from . import nn_parts as _nn

class _Param(_paramRange.VirtualParam):
    def __init__(self):
        self.model_name = 'LSTM+drop+FC'
        ParamItem = _paramRange.ParamItem
        self.parameters = {
                'input_dim':ParamItem('input_dimension',40,False),
                'label_dim':ParamItem('label_dimension',2,False),
                'temporal_len':ParamItem('temporal_length',30,False),
                'hs':ParamItem('hidden_size',128,False),
                'layer_num':ParamItem('layer_number',1,False),
                'dp':ParamItem('dropout_rate',0.5,True),
                'dp_flag':ParamItem('dropout_flag',0,False),
                'ln_flag':ParamItem('layer_norm_flag',0,False),
                'res_flag':ParamItem('residual_learning_flag',0,False),
                'gated_flag':ParamItem('gated_activation_flag',0,False),
                'peephole_flag':ParamItem('lstm_peephole_flag',1,False),
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
                'label_dim':ParamRangeItem(2,2,False),
                'temporal_len':ParamRangeItem(10,200,False),
                'hs':ParamRangeItem(16,1024,False),
                'layer_num':ParamRangeItem(1,6,False),
                'dp':ParamRangeItem(0.2,0.8,False),
                'ln_flag':ParamRangeItem(0,2,False),
                'res_flag':ParamRangeItem(0,2,False),
                'skip_flag':ParamRangeItem(0,2,False),
                'gated_flag':ParamRangeItem(0,2,False),
                'dp_flag':ParamRangeItem(0,2,False),
                'peephole_flag':ParamRangeItem(0,2,False),
                'l2':ParamRangeItem(-7,0,True),
                'lr':ParamRangeItem(-7,0,True),
                'batch_size':ParamRangeItem(16,16,False),
                'epoch_num':ParamRangeItem(100000,100000,False),
                'min_delta':ParamRangeItem(1e-3,1e-3,False),
                'patience':ParamRangeItem(3000,3000,False),
                }

def _linear_activation(x):
    return x

def _get_first_half_channel(x,name='first_half_channel'):
    x_shape = x.shape.as_list()
    rank = len(x_shape)
    channel_num = x_shape[-1]
    return _tf.slice(x,[0]*rank,[-1]*(rank-1)+[int(channel_num/2)])

def _get_last_half_channel(x,name='last_half_channel'):
    x_shape = x.shape.as_list()
    rank = len(x_shape)
    channel_num = x_shape[-1]
    return _tf.slice(x,[0]*(rank-1)+[int(channel_num/2)],[-1]*rank)


class Model(_virtualNeuralNetwork.VirtualNN):
    def lstm_block(self,input_tensor,name_scope='LSTMBlock'):
        param = self.param
        with _tf.name_scope(name_scope):
            lstm_layer = _nn.rnn.LSTMCell(input_tensor.shape.as_list()[-1],param.hs*(2 if param.gated_flag else 1),name_scope='LSTMLayer')
            lstm_out = lstm_layer(input_tensor)
            if param.gated_flag:
                lstm_out_candidate = _get_first_half_channel(lstm_out)
                if param.ln_flag:
                    lstm_candidate_ln_layer = _nn.norm.LayerNorm([param.hs],name_scope='lstm_candidate_layer_norm_layer')
                    lstm_out_candidate = lstm_candidate_ln_layer(lstm_out_candidate)
                lstm_out_candidate = _tf.tanh(lstm_out_candidate)
                lstm_out_gate = _get_last_half_channel(lstm_out)
                if param.ln_flag:
                    lstm_gate_ln_layer = _nn.norm.LayerNorm([param.hs],name_scope='lstm_gate_layer_norm_layer')
                    lstm_out_gate = lstm_gate_ln_layer(lstm_out_gate)
                lstm_out_gate = _tf.sigmoid(lstm_out_gate)
                lstm_out = _tf.multiply(lstm_out_candidate,lstm_out_gate,name='lstm_gated_out')
            else:
                if param.ln_flag:
                    lstm_ln_layer = _nn.norm.LayerNorm([param.hs],name_scope='lstm_layer_norm_layer')
                    lstm_out = lstm_ln_layer(lstm_out)
                lstm_out = _tf.nn.relu(lstm_out,name='lstm_relu_out')
            if param.dp_flag:
                lstm_out = _tf.layers.dropout(lstm_out,param.dp,training=self.train_flag,name='lstm_out_drop')
        return lstm_layer,lstm_out

    def build_model(self):
        param = self.param
        with self.graph.as_default():
            with _tf.name_scope('Input'):
                self.input_data = _tf.placeholder(_tf.float32,[None,None,param.input_dim],name='input_data')
                self.input_label = _tf.placeholder(_tf.float32,[None,None,param.label_dim],name='label')
                self.train_flag = _tf.placeholder(_tf.bool,name='train_flag')

            self.stacked_lstm_layers = []
            self.stacked_lstm_input = [self.input_data]
            self.stacked_lstm_output = []
            for layer_num in range(1,param.layer_num+1):
                lstm_layer,lstm_out = self.lstm_block(self.stacked_lstm_input[-1],name_scope='LSTMBlock%d'%layer_num)
                self.stacked_lstm_layers.append(lstm_layer)
                self.stacked_lstm_output.append(lstm_out)
                if param.res_flag and layer_num > 1:
                    self.stacked_lstm_input.append(_tf.add(self.stacked_lstm_input[-1],lstm_out,name='lstm_residual_out%d'%layer_num))
                else:
                    self.stacked_lstm_input.append(lstm_out)
            if param.skip_flag:
                self.stacked_lstm_out = _tf.nn.relu(_tf.add_n(self.stacked_lstm_output),name='stacked_lstm_out')
            else:
                self.stacked_lstm_out = self.stacked_lstm_input[-1]

            with _tf.name_scope('Flat'):
                self.lstm_out_drop_flat = _tf.reshape(self.stacked_lstm_out,[-1,param.hs],name='lstm_out_drop_flat')
                self.input_label_flat= _tf.reshape(self.input_label,[-1,param.label_dim],name='input_label_flat')

            with _tf.name_scope('Dense'):
                self.dense_layer = _nn.fc.Dense(param.hs,param.label_dim,activation_func=_linear_activation,name_scope='DenseLayer')
                self.logits = self.dense_layer(self.lstm_out_drop_flat)

            with _tf.name_scope('Loss'):
                self.cross_entropy_loss = _tf.reduce_mean(_tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label_flat,logits=self.logits,dim=-1),name='softmax_loss')
                self.l2_loss = _tf.reduce_sum([self.dense_layer.get_l2_loss()] + [l.get_l2_loss() for l in self.stacked_lstm_layers],name='l2_loss')
                self.loss = _tf.add(self.cross_entropy_loss,param.l2*self.l2_loss,name='loss')
                self.adam = _tf.train.AdamOptimizer(param.lr).minimize(self.loss)

            with _tf.name_scope('Accuracy'):
                self.accuracy = _tf.reduce_mean(_tf.cast(_tf.equal(_tf.argmax(self.input_label_flat,axis=1),_tf.argmax(self.logits,axis=1)),_tf.float32),name='accuracy')

