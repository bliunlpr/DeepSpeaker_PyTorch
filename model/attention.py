import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
from collections import OrderedDict
import math
import os
import six
import numpy as np
import torch.utils.model_zoo as model_zoo

from model.base_model import SequenceWise


class AttentionModel(nn.Module):
    def __init__(self, opt):
        super(AttentionModel, self).__init__()
        self.att_type = opt.att_type
                                      
        if self.att_type == 'last_state':
            self.model = LastState()
        elif self.att_type == 'average_state':
            self.model = MeanState()
        elif self.att_type == 'base_attention':
            self.model = BaseAtt(opt.embedding_size)
        elif self.att_type == 'multi_attention':
            self.model = MultiAtt(opt.embedding_size, opt.attention_dim, opt.attention_head_num)
        else:
            raise AssertionError("att_type {} not supported!".format(self.att_type))
                        
    def forward(self, x):
        x, attn = self.model(x)
        return x, attn


class SeqAttentionModel(nn.Module):
    def __init__(self, opt):
        super(SeqAttentionModel, self).__init__()
        self.att_type = opt.att_type
        self.segment_type = opt.segment_type
          
        if self.att_type == 'last_state':
            self.model = LastState()
        elif self.att_type == 'average_state':
            self.model = MeanState()
        elif self.att_type == 'base_attention':
            self.model = BaseAtt(opt.embedding_size)
        elif self.att_type == 'multi_attention':
            self.model = MultiAtt(opt.embedding_size, opt.attention_dim, opt.attention_head_num)
        elif self.att_type == 'divide_attention':
            self.model = DivideAtt(opt.embedding_size, opt.attention_dim, opt.attention_head_num)    
        else:
            raise AssertionError("att_type {} not supported!".format(self.att_type))
                        
    def forward(self, x, segment_num):
        assert x.size(0) == torch.sum(segment_num), print(x.size(), torch.sum(segment_num))   
        out = None
        attn_i = None
        attn = None
        out_segment = None
        start = 0
        for i in range(segment_num.size(0)):
            if segment_num[i] <= 0:
              continue
            end = start + int(segment_num[i])
            out_i = x[start:end, :]
            start += int(segment_num[i])
            if len(out_i.shape) == 1:
                out_i = out_i.unsqueeze(0) 
                 
            if self.segment_type == 'average':
                out_segment_i = torch.mean(out_i, dim=0, keepdim=True)
            elif self.segment_type == 'all':
                out_segment_i = out_i
            else:
                out_segment_i = out_i 
            
            out_i, attn_i = self.model(out_i)                                                                 
            out_i = out_i.unsqueeze(0)
            if out is None:
                out = out_i
            else:
                out = torch.cat((out, out_i), 0)
                                 
            if attn_i is not None:
                attn_i = torch.mm(attn_i.transpose(0, 1), attn_i).unsqueeze(0)
                if attn is None:
                    attn = attn_i
                else:
                    attn = torch.cat((attn, attn_i), 0) 
                    
            if out_segment_i is not None:
                if out_segment is None:
                    out_segment = out_segment_i
                else:
                    out_segment = torch.cat((out_segment, out_segment_i), 0)
                                                   
        return out, out_segment, attn
        
                
class LastState(nn.Module):
    def __init__(self):
        super(LastState, self).__init__()
        
    def forward(self, x):
        x = x[-1]
        return x, None


class MeanState(nn.Module):
    def __init__(self):
        super(MeanState, self).__init__()
        
    def forward(self, x):
         x = torch.mean(x, dim=0)
         return x, None

                   
class BaseAtt(nn.Module):
    def __init__(self, input_size):
        super(BaseAtt, self).__init__()
        self.query = nn.Linear(input_size, 1)        
    def forward(self, x):   
        if len(x.size()) == 3:
            value = x.transpose(0, 1).transpose(1, 2)
            t, n = x.size(0), x.size(1)
            x = x.view(t * n, -1)
            query = self.query(x)
            query = query.view(t, n, -1).transpose(0, 1)
            query = F.softmax(query, 1)
            out = torch.bmm(value, query).squeeze(2)
        elif len(x.size()) == 2:         
            query = self.query(x).squeeze(1) 
            query = F.softmax(query, 0)
            out = x.transpose(0, 1)
            out = torch.mv(out, query)                 
        return out, None
        
            
class MultiAtt(nn.Module):
    def __init__(self, input_size, attention_dim, attention_head_num):
        super(MultiAtt, self).__init__()
        self.w1 = SequenceWise(nn.Sequential(nn.Linear(input_size, attention_dim, bias=False),
                                             nn.ReLU(),
                                             nn.Linear(attention_dim, attention_head_num, bias=False)))
        self.post_fc = nn.Linear(input_size * attention_head_num, input_size, bias=False)
        
    def forward(self, x):
        if len(x.size()) == 3:
            attn = self.w1(x)
            attn = F.softmax(attn, 0).transpose(0, 1)
            x = torch.bmm(x.transpose(0, 1).transpose(1, 2), attn) 
            out = x.view(x.size(0), -1)              
        elif len(x.size()) == 2:
            attn = self.w1(x)
            attn = F.softmax(attn, 0)
            out = torch.mm(x.transpose(0, 1), attn) 
            out = out.view(-1)
        out = self.post_fc(out) 
        return out, attn
        
                    
class DivideAtt(nn.Module):
    def __init__(self, input_size, attention_dim, attention_head_num):
        super(DivideAtt, self).__init__()
        self.input_size = input_size
        self.w1 = SequenceWise(nn.Sequential(nn.Linear(input_size, attention_dim, bias=False),
                                             nn.ReLU(),
                                             nn.Linear(attention_dim, attention_head_num, bias=False)))
        self.post_fc = nn.Linear(input_size * attention_head_num, input_size, bias=False)
    def forward(self, x):
        if len(x.size()) == 3:
            x_a = x[:, :, :self.input_size]
            x_b = x[:, :, self.input_size:]
            attn = self.w1(x_b)
            attn = F.softmax(attn, 0).transpose(0, 1)
            x = torch.bmm(x_a.transpose(0, 1).transpose(1, 2), attn) 
            out = x.view(x.size(0), -1)  
        elif len(x.size()) == 2:
            x_a = x[:, :self.input_size]
            x_b = x[:, self.input_size:]
            attn = self.w1(x_b)
            attn = F.softmax(attn, 0)
            out = torch.mm(x_a.transpose(0, 1), attn) 
            out = out.view(-1)      
        out = self.post_fc(out) 
                  
        return out, attn      
        