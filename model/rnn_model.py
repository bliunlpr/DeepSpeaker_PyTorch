import math
import numpy as np
import torch
import torch.nn as nn

from model.base_model import ModelBase
from model.attention import AttentionModel, SeqAttentionModel
from model.rnn_backbone.rnn_network import LSTM, LSTMP, BLSTMP

device = torch.device("cuda")
supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

                       
class DeepSpeakerRNNModel(ModelBase):
    def __init__(self, opt):
        super(DeepSpeakerRNNModel, self).__init__(opt)        
        self._input_size = opt.in_size
        self._hidden_size = opt.rnn_hidden_size
        self._hidden_layers = opt.rnn_layers
        self._rnn_type = supported_rnns[opt.rnn_type]
        self._bidirectional = opt.rnn_bidirectional
        self._dropout = opt.dropout
        self.att_type = opt.att_type
                        
        if opt.loss_type.split('_')[1] == 'softmax' or opt.loss_type.split('_')[1] == 'contrast':
            self.w = nn.Parameter(torch.FloatTensor(np.array([10])))
            self.b = nn.Parameter(torch.FloatTensor(np.array([-5])))
        else:
            self.w = None
            self.b = None
        
        if self.model_type == 'lstm':
            self.rnns = LSTM(self._rnn_type, self._input_size, self._hidden_layers, self._hidden_size, self.embedding_size, self._bidirectional, self._dropout)          
        elif self.model_type == 'blstmp':
            self.rnns = BLSTMP(self._input_size, self._hidden_layers, self._hidden_size, self.embedding_size, opt.dropout)                
        elif self.model_type == 'lstmp':
            self.rnns = LSTMP(self._input_size, self._hidden_layers, self._hidden_size, self.embedding_size, opt.dropout)                
        
        self.att = AttentionModel(opt)
         
    def forward(self, x, segment_num=None):
        if len(x.size()) > 3:
            x = x.squeeze(0)
        elif len(x.size()) == 3:
            x = x.transpose(0, 1)
        x = self.rnns(x)
        x, attn = self.att(x)             
        out = self.normalize(x)
        
        if not self.training:
            return out, self.w, self.b
        else:
            return out, attn, self.w, self.b   


class DeepSpeakerRNNSeqModel(DeepSpeakerRNNModel):
    def __init__(self, opt):
        super(DeepSpeakerRNNSeqModel, self).__init__(opt)            
        self.segment_type = opt.segment_type                                                
        self.att = SeqAttentionModel(opt)
        
    def forward(self, x, segment_num):
        if len(x.size()) > 3:
            x = x.squeeze(0)
        elif len(x.size()) == 3:
            x = x.transpose(0, 1)
        x = self.rnns(x) 
        x = x[-1]
        out, out_segment, attn = self.att(x, segment_num)      
                    
        out = self.normalize(out)
        
        if out_segment is not None:
            out_segment = self.normalize(out_segment)
            
        if not self.training:
            return out, self.w, self.b
        else:
            return out, out_segment, attn, self.w, self.b
            
    