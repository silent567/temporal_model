from . import paramRange as _paramRange
from . import virtualNeuralNetwork as _virtualNeuralNetwork
import tensorflow as _tf
from . import nn_parts as _nn

class _Param(_paramRange.VirtualParam):
    def __init__(self):
        self.model_name = 'cconv+stacked_cdconv+FC'
        ParamItem = _paramRange.ParamItem
        self.parameters = {
                'input_dim':ParamItem('input_dimension',40,False),
                'label_dim':ParamItem('label_dimension',2,False),
                'temporal_len':ParamItem('temporal_length',1024,False),
                'cdconv_layer_num':ParamItem('causal_dilated_conv_layer_number',5,False),
                'cdconv_hs':ParamItem('causal_dilated_conv_hidden_size',128,False),
                'hs':ParamItem('hidden_size',32,False),
                'ln_flag':ParamItem('layer_norm_flag',0,False),
                'res_flag':ParamItem('residual_learning_flag',1,False),
                'skip_flag':ParamItem('skip_connections_flag',1,False),
                'gated_flag':ParamItem('gated_activation_flag',0,False),
                'dp_flag':ParamItem('dropout_flag',1,False),
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
                'label_dim':ParamRangeItem(2,2,False),
                'temporal_len':ParamRangeItem(1024,1024,False),
                'cdconv_layer_num':ParamRangeItem(1,8,False),
                'cdconv_hs':ParamRangeItem(16,1024,False),
                'hs':ParamRangeItem(16,1024,False),
                'ln_flag':ParamRangeItem(0,2,False),
                'res_flag':ParamRangeItem(0,2,False),
                'skip_flag':ParamRangeItem(0,2,False),
                'gated_flag':ParamRangeItem(0,2,False),
                'dp_flag':ParamRangeItem(0,2,False),
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
    def _causal_dilated_conv_layer(self,input_tensor,train_flag,dilation_rate,name_scope='CausalDilatedConvLayer'):
        param = self.param
        with _tf.name_scope(name_scope):
            cdconv_layer = _nn.conv.CausalDilatedConvCell([2,1,input_tensor.shape.as_list()[-1],param.cdconv_hs*(2 if param.gated_flag else 1)],[dilation_rate,1],name_scope='cdconv_layer')
            cdconv_out = cdconv_layer(input_tensor)
            if param.gated_flag:
                cdconv_out_candidate = _get_first_half_channel(cdconv_out)
                if param.ln_flag:
                    cdconv_candidate_ln_layer = _nn.norm.LayerNorm([param.input_dim,param.cdconv_hs],name_scope='cdconv_candidate_layer_norm_layer')
                    cdconv_out_candidate = cdconv_candidate_ln_layer(cdconv_out_candidate)
                cdconv_out_candidate = _tf.tanh(cdconv_out_candidate)
                cdconv_out_gate = _get_last_half_channel(cdconv_out)
                if param.ln_flag:
                    cdconv_gate_ln_layer = _nn.norm.LayerNorm([param.input_dim,param.cdconv_hs],name_scope='cdconv_gate_layer_norm_layer')
                    cdconv_out_gate = cdconv_gate_ln_layer(cdconv_out_gate)
                cdconv_out_gate = _tf.sigmoid(cdconv_out_gate)
                cdconv_out = _tf.multiply(cdconv_out_candidate,cdconv_out_gate,name='cdconv_gated_out')
            else:
                if param.ln_flag:
                    cdconv_ln_layer = _nn.norm.LayerNorm([param.input_dim,param.cdconv_hs],name_scope='cdconv_layer_norm_layer')
                    cdconv_out = cdconv_ln_layer(cdconv_out)
                cdconv_out = _tf.nn.relu(cdconv_out,name='cdconv_relu_out')
            if param.dp_flag:
                cdconv_out = _tf.layers.dropout(cdconv_out,param.dp,training=train_flag,name='cdconv_out_drop')
        return cdconv_layer,cdconv_out

    def build_model(self):
        param = self.param
        with self.graph.as_default():
            with _tf.name_scope('Input'):
                self.input_data = _tf.placeholder(_tf.float32,[None,None,param.input_dim],name='input_data')
                self.input_label = _tf.placeholder(_tf.float32,[None,None,param.label_dim],name='label')
                self.train_flag = _tf.placeholder(_tf.bool,name='train_flag')

            with _tf.name_scope('ReshapeInput'):
                self.input_data_exp = _tf.expand_dims(self.input_data,axis=-1,name='input_data_exp')
            self.cdconv_layer,self.cdconv_out = self._causal_dilated_conv_layer(self.input_data_exp,self.train_flag,1,name_scope='CausalConvLayer')
            # print(self.cdconv_out.shape)
            with _tf.name_scope('StackedCausalDilatedConv'):
                self.stacked_cdconv_layers = []
                self.stacked_cdconv_input = [self.cdconv_out]
                self.stacked_cdconv_output = []
                for layer_num in range(1,param.cdconv_layer_num+1):
                    cdconv_layer,cdconv_out = self._causal_dilated_conv_layer(self.stacked_cdconv_input[-1],self.train_flag,int(2**layer_num),name_scope='StackedCausalDilatedConvLayer%d'%layer_num)
                    # print(cdconv_out.shape)
                    self.stacked_cdconv_layers.append(cdconv_layer)
                    self.stacked_cdconv_output.append(cdconv_out)
                    if param.res_flag:
                        self.stacked_cdconv_input.append(_tf.add(self.stacked_cdconv_input[-1],cdconv_out,name='cdconv_residual_out%d'%layer_num))
                    else:
                        self.stacked_cdconv_input.append(cdconv_out)
                if param.skip_flag:
                    self.stacked_cdconv_out = _tf.nn.relu(_tf.add_n([self.cdconv_out]+self.stacked_cdconv_output),name='stacked_cdconv_out')
                else:
                    self.stacked_cdconv_out = self.stacked_cdconv_input[-1]
                # print(self.stacked_cdconv_out.shape)

            with _tf.name_scope('Flat'):
                self.stacked_cdconv_out_flat = _tf.reshape(self.stacked_cdconv_out,[-1,param.cdconv_hs*param.input_dim],name='stacked_cdconv_out_flat')
                self.input_label_flat= _tf.reshape(self.input_label,[-1,param.label_dim],name='input_label_flat')
                # print(self.stacked_cdconv_out_flat.shape,self.input_label_flat.shape)

            with _tf.name_scope('Dense'):
                self.dense_layer = _nn.fc.Dense(param.cdconv_hs*param.input_dim,param.label_dim,activation_func=_linear_activation,name_scope='DenseLayer')
                self.logits = self.dense_layer(self.stacked_cdconv_out_flat)
                # print(self.logits.shape)

            with _tf.name_scope('Loss'):
                self.cross_entropy_loss = _tf.reduce_mean(_tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label_flat,logits=self.logits,dim=-1),name='softmax_loss')
                self.l2_loss = _tf.add_n([self.dense_layer.get_l2_loss(),self.cdconv_layer.get_l2_loss()]+[l.get_l2_loss() for l in self.stacked_cdconv_layers],name='l2_loss')
                self.loss = _tf.add(self.cross_entropy_loss,param.l2*self.l2_loss,name='loss')
                self.adam = _tf.train.AdamOptimizer(param.lr).minimize(self.loss)

            with _tf.name_scope('Accuracy'):
                self.accuracy = _tf.reduce_mean(_tf.cast(_tf.equal(_tf.argmax(self.input_label_flat,axis=1),_tf.argmax(self.logits,axis=1)),_tf.float32),name='accuracy')

