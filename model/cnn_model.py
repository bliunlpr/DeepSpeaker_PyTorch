import math
import numpy as np

import torch
import torch.nn as nn

from model.base_model import ModelBase
from model.attention import AttentionModel, SeqAttentionModel
from model.cnn_backbone.mobilefacenet import MobileFaceNet
from model.cnn_backbone.cbam import CBAMResNet
from model.cnn_backbone.attention_net import ResidualAttentionNet_56, ResidualAttentionNet_92


class DeepSpeakerCNNModel(ModelBase):
    def __init__(self, opt):
        super(DeepSpeakerCNNModel, self).__init__(opt)
        
        # define backbone layer
        if self.model_type == 'cnn_MobileFace':
            self.model = MobileFaceNet()
        elif self.model_type == 'cnn_Res50_IR':
            self.model = CBAMResNet(50, feature_dim=self.embedding_size, mode='ir')
        elif self.model_type == 'cnn_SERes50_IR':
            self.model = CBAMResNet(50, feature_dim=self.embedding_size, mode='ir_se')
        elif self.model_type == 'cnn_Res100_IR':
            self.model = CBAMResNet(100, feature_dim=self.embedding_size, mode='ir')
        elif self.model_type == 'cnn_SERes100_IR':
            self.model = CBAMResNet(100, feature_dim=self.embedding_size, mode='ir_se')
        elif self.model_type == 'cnn_Attention_56':
            self.model = ResidualAttentionNet_56(feature_dim=self.embedding_size)
        elif self.model_type == 'cnn_Attention_92':
            self.model = ResidualAttentionNet_92(feature_dim=self.embedding_size)
        else:
            print(self.model_type, ' is not available!')
        
        if opt.loss_type.split('_')[1] == 'softmax' or opt.loss_type.split('_')[1] == 'contrast':
            self.w = nn.Parameter(torch.FloatTensor(np.array([10])))
            self.b = nn.Parameter(torch.FloatTensor(np.array([-5])))
        else:
            self.w = None
            self.b = None
    
    def forward(self, x, segment_num=None):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        elif len(x.size()) == 4:
            x = x.transpose(1, 2).transpose(0, 1)
        x = self.model(x)             
        out = self.normalize(x)
        
        if not self.training:
            return out, self.w, self.b
        else:
            return out, None, self.w, self.b

        
class DeepSpeakerCNNSeqModel(DeepSpeakerCNNModel):
    def __init__(self, opt):
        super(DeepSpeakerCNNSeqModel, self).__init__(opt)
        self.segment_type = opt.segment_type                      
        self.att = SeqAttentionModel(opt)            
                    
    def forward(self, x, segment_num):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        elif len(x.size()) == 4:
            x = x.transpose(1, 2).transpose(0, 1)
        x = self.model(x) 
        out, out_segment, attn = self.att(x, segment_num)      
                    
        out = self.normalize(out)
        
        if out_segment is not None:
            out_segment = self.normalize(out_segment)
            
        if not self.training:
            return out, self.w, self.b
        else:
            return out, out_segment, attn, self.w, self.b
            
            