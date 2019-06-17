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

device = torch.device("cuda")
supported_acts = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(), 'tanh': nn.Tanh(), 
                  'leakyrelu': nn.LeakyReLU(), 'prelu': nn.PReLU()}


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) > 2:
            t, n = x.size(0), x.size(1)
            x = x.view(t * n, -1)
            x = self.module(x)
            x = x.view(t, n, -1)
        else:
            x = self.module(x)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr
        

class ModelBase(nn.Module):
    """
    ModelBase class for sharing code among various model.
    """
    def __init__(self, opt):
        super(ModelBase, self).__init__()
        self.opt = opt
        self.model_type = opt.model_type
        self.embedding_size = opt.embedding_size
        
        self.margin_s = opt.margin_s
        self.margin_m = opt.margin_m    
        self.cos_m = math.cos(self.margin_m)
        self.sin_m = math.sin(self.margin_m)
        self.th = math.cos(math.pi - self.margin_m)
        self.mm = math.sin(math.pi - self.margin_m) * self.margin_m
        
    def forward(self, x):
        raise NotImplementedError
    
    @classmethod
    def load_model(cls, path, state_dict, model_opt):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(opt=model_opt)
        if package[state_dict] is not None:
            model_load(package, 'state_dict', model) 
        return model
    
    @staticmethod
    def serialize(model, state_dict, optimizer=None, optim_dict=None):
        model_is_cuda = next(model.parameters()).is_cuda
        model = model.module if model_is_cuda else model
        package = {
            'opt': model.opt,
            'state_dict': model.state_dict()            
        }
        if optimizer is not None:
            package[optim_dict] = optimizer.state_dict()
        return package
        
    @staticmethod    
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
        
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features  
    
    def set_margin_s(self, margin_s):
        self.margin_s = margin_s
    
    def similarity(self, embedded, w, b, center=None):
        """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
            Input center to test enrollment. (embedded for verification)
        :return: tf similarity matrix (NM x N)
        """
        N = self.opt.speaker_num
        M = self.opt.utter_num 
        P = self.opt.embedding_size
        ##S = opt.segment_num
        '''if self.opt.train_type == 'multi_attention' or self.opt.train_type == 'divide_attention':
            P = self.opt.embedding_size * self.opt.attention_head_num   
        else: 
            P = self.opt.embedding_size'''
        ##embedded_mean = torch.cat([torch.mean(embedded[i*S:(i+1)*S,:], dim=0, keepdim=True) for i in range(N*M)], dim=0)
        embedded_split = torch.reshape(embedded, (N, M, P))
    
        if center is None:
            center = self.normalize(torch.mean(embedded_split, dim=1))              # [N,P] normalized center vectors eq.(1)
            center_except = self.normalize(torch.reshape(torch.sum(embedded_split, dim=1, keepdim=True)
                                                 - embedded_split, (N*M,P)))  # [NM,P] center vectors eq.(8)
            # make similarity matrix eq.(9)
            S = torch.cat(
                [torch.cat([torch.sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], dim=1, keepdim=True) if i==j
                            else torch.sum(center[i:(i+1),:]*embedded_split[j,:,:], dim=1, keepdim=True) for i in range(N)],
                           dim=1) for j in range(N)], dim=0)
        else :
            # If center(enrollment) exist, use it.
            S = torch.cat(
                [torch.cat([torch.sum(center[i:(i + 1), :] * embedded_split[j, :, :], dim=1, keepdim=True) for i
                            in range(N)], dim=1) for j in range(N)], dim=0)
        
        if self.opt.loss_type.split('_')[1] == 'softmax' or self.opt.loss_type.split('_')[1] == 'contrast':
            S = torch.abs(w)*S + b   # rescaling
    
        return S
    
    def similarity_segment(self, embedded, seq_len, w, b, center=None):
        """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
            Input center to test enrollment. (embedded for verification)
        :return: tf similarity matrix (NM x N) normalize(
        """
        N = self.opt.speaker_num
        M = self.opt.utter_num
        ##assert embedded.size(0) == torch.sum(seq_len), print(embedded.size(), seq_len)
    
        s, seq_index, seq_utter_index = 0, [], []
        for i in range(seq_len.size(0)):
            seq_index.append(s)
            if i % M == 0:
                seq_utter_index.append(s)
            s += int(seq_len[i])
        seq_index.append(s)
        seq_utter_index.append(s)
    
        center = self.normalize(
            torch.cat([torch.mean(embedded[seq_utter_index[i]:seq_utter_index[i + 1], :], dim=0, keepdim=True) for i
                       in range(len(seq_utter_index) - 1)], dim=0))
    
        center_except = self.normalize(torch.cat([(torch.sum(embedded[seq_utter_index[i]:seq_utter_index[i + 1], :], dim=0,keepdim=True)
                                              - embedded[seq_utter_index[i]:seq_utter_index[i + 1],:]) / (seq_utter_index[i + 1] - seq_utter_index[i] - 1)
                                             for i in range(len(seq_utter_index) - 1)], dim=0))
    
        S = torch.cat(
            [torch.cat(
                [torch.sum(center_except[seq_utter_index[i]:seq_utter_index[i + 1], :] * embedded[seq_utter_index[j]:seq_utter_index[j + 1], :],
                           dim=1, keepdim=True) if i == j else torch.sum(center[i:(i + 1), :] * embedded[seq_utter_index[j]:seq_utter_index[j + 1], :],
                           dim=1, keepdim=True) for i in range(N)], dim=1) for j in range(N)], dim=0)
        
        if self.opt.loss_type == 'seq_softmax' or self.opt.loss_type == 'seq_contrast':
            S = torch.abs(w) * S + b  # rescaling
    
        return S
        
    def normalize(self, x):
        """ normalize the last dimension vector of the input matrix .unsqueeze(0)
        :return: normalized input
        """
        return x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + 1e-6)     
    
    def loss_cal(self, S):
        """ calculate loss with similarity matrix(S) eq.(6) (7) 
        :type: "softmax" or "contrast"
        :return: loss
        """
        N = self.opt.speaker_num
        M = self.opt.utter_num 
        loss_type = self.opt.loss_type
        S_correct = torch.cat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], dim=0)  # colored entries in Fig.1
    
        if loss_type == "ge2e_softmax" or loss_type == "seq_softmax":
            total = -torch.sum(S_correct-torch.log(torch.sum(torch.exp(S), dim=1, keepdim=True) + 1e-6))
        elif loss_type == "ge2e_cosine_margin" or loss_type == "seq_cosine_margin":
            S_correct_scale = torch.cat([S[i*M:(i+1)*M, i:(i+1)] * self.margin_s - self.margin_m for i in range(N)], dim=0)  # colored entries in Fig.1
            S_all_scale = torch.cat([torch.cat([S[i*M:(i+1)*M, i:(i+1)] * self.margin_s - self.margin_m if i==j
                                               else S[i*M:(i+1)*M, j:(j+1)] * self.margin_s for j in range(N)], dim=1)
                                               for i in range(N)], dim=0)
            total = -torch.sum(S_correct_scale-torch.log(torch.sum(torch.exp(S_all_scale), dim=1, keepdim=True) + 1e-6))
        elif loss_type == "ge2e_contrast" or loss_type == "seq_contrast":
            S_sig = torch.sigmoid(S)
            S_sig = torch.cat([torch.cat([0*S_sig[i*M:(i+1)*M, j:(j+1)] if i==j
                                  else S_sig[i*M:(i+1)*M, j:(j+1)] for j in range(N)], dim=1)
                                 for i in range(N)], dim=0)
            total = torch.sum(1-torch.sigmoid(S_correct)+torch.max(S_sig, dim=1, keepdim=True)[0])
        else:
            raise AssertionError("loss type should not be {} !".format(loss_type))
        ##total = total / (N * M * N)
        return total
    
    def loss_cal_segment(self, S, seq_len):
        """ calculate loss with similarity matrix(S) eq.(6) (7)
        :type: "softmax" or "contrast"
        :return: loss
        """
        N = self.opt.speaker_num
        M = self.opt.utter_num
        loss_type = self.opt.loss_type
    
        ##assert S.size(0) == torch.sum(seq_len), print(S.size(), seq_len)
        ##assert N * M == seq_len.size(0), print(N, M, seq_len)
    
        s, seq_index, seq_utter_index = 0, [], []
        for i in range(seq_len.size(0)):
            seq_index.append(s)
            if i % M == 0:
                seq_utter_index.append(s)
            s += int(seq_len[i])
        seq_index.append(s)
        seq_utter_index.append(s)
    
        S_correct = torch.cat([S[seq_utter_index[i]:seq_utter_index[i + 1], i:(i + 1)] for i in range(N)],
                              dim=0)  # colored entries in Fig.1
    
        if loss_type == "seq_softmax":
            total = -torch.sum(S_correct - torch.log(torch.sum(torch.exp(S), dim=1, keepdim=True) + 1e-6))
        elif loss_type == "seq_cosine_margin":
            S_correct_scale = torch.cat([S[seq_utter_index[i]:seq_utter_index[i + 1], i:(i + 1)] * self.margin_s - self.margin_m for i in range(N)], dim=0) 
            S_all_scale = torch.cat([torch.cat([S[seq_utter_index[i]:seq_utter_index[i + 1], i:(i+1)] * self.margin_s - self.margin_m if i==j
                                               else S[seq_utter_index[i]:seq_utter_index[i + 1], j:(j+1)] * self.margin_s for j in range(N)], dim=1)
                                               for i in range(N)], dim=0)
            total = -torch.sum(S_correct_scale - torch.log(torch.sum(torch.exp(S_all_scale), dim=1, keepdim=True) + 1e-6))
        elif loss_type == "seq_contrast":
            S_sig = torch.sigmoid(S)
            S_sig = torch.cat([torch.cat([0 * S_sig[i * M:(i + 1) * M, j:(j + 1)] if i == j
                                          else S_sig[i * M:(i + 1) * M, j:(j + 1)] for j in range(N)], dim=1)
                               for i in range(N)], dim=0)
            total = torch.sum(1 - torch.sigmoid(S_correct) + torch.max(S_sig, dim=1, keepdim=True)[0])
        else:
            raise AssertionError("loss type should be softmax or contrast !")
    
        total = total * int(seq_len.size(0)) / float(torch.sum(seq_len))
        ##total = total / (N * M * N)
        return total
        
    def penalty_loss_cal(self, A):
        loss_call = torch.nn.MSELoss(reduction='sum')
        
        I = torch.eye(A.size(2)).to(self.opt.device)
        out = torch.bmm(A.transpose(1, 2), A)
        return loss_call(out, I.expand_as(out))
        
    def penalty_seq_loss_cal(self, A):
        #print(A.shape)
        loss_call = torch.nn.MSELoss(reduction='sum')
        I = torch.eye(A.size(1)).to(self.opt.device)
        return loss_call(A, I.expand_as(A))
