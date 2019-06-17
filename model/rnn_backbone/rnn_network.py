import torch
import torch.nn as nn
from collections import OrderedDict

from model.base_model import SequenceWise


class BLSTMP(torch.nn.Module):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTMP, self).__init__()
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            setattr(self, "bilstm%d" % i, torch.nn.LSTM(inputdim, cdim, dropout=dropout,
                                                        num_layers=1, bidirectional=True))
            # bottleneck layer to merge
            setattr(self, "bproject%d" % i, torch.nn.Linear(2 * cdim, hdim))
            
        self.elayers = elayers
        self.cdim = cdim
       
    def forward(self, xpad):
        '''BLSTMP forward
        :param xs:
        :param ilens:
        :return:
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            bilstm = getattr(self, 'bilstm' + str(layer))
            ys, (hy, cy) = bilstm(xpad)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bproject' + str(layer)
                                )(ys.contiguous().view(-1, ys.size(2)))
            xpad = torch.tanh(projected.view(ys.size(0), ys.size(1), -1))
            del hy, cy

        return xpad  # x: utt list of frame x dim
        

class LSTMP(torch.nn.Module):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(LSTMP, self).__init__()
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            setattr(self, "lstm%d" % i, torch.nn.LSTM(inputdim, cdim, dropout=dropout,
                                                      num_layers=1, bidirectional=False))
            # bottleneck layer to merge
            setattr(self, "project%d" % i, torch.nn.Linear(cdim, hdim))
        
        self.elayers = elayers
        self.cdim = cdim
       
    def forward(self, xpad):
        '''BLSTMP forward
        :param xs:
        :param ilens:
        :return:
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            bilstm = getattr(self, 'lstm' + str(layer))
            ys, (hy, cy) = bilstm(xpad)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'project' + str(layer)
                                )(ys.contiguous().view(-1, ys.size(2)))
            xpad = torch.tanh(projected.view(ys.size(0), ys.size(1), -1))
            del hy, cy

        return xpad  # x: utt list of frame x dim
        

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, dropout=0, batch_norm=False):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        ##self.rnn.flatten_parameters()
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        if self.dropout is not None:
            x = self.dropout(x)
        return x
        
        
class LSTM(torch.nn.Module):
    def __init__(self, rnn_type, idim, elayers, hdim, odim, bidirectional, dropout):
        super(LSTM, self).__init__()
        
        rnns = []
        for i in range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            if i == elayers - 1:
                dropout = 0
            else:
                dropout = dropout
                
            rnn = BatchRNN(input_size=inputdim, hidden_size=hdim, rnn_type=rnn_type,
                           bidirectional=bidirectional, dropout=dropout, batch_norm=False)
            rnns.append(('%d' % (i), rnn))
            
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.fc = SequenceWise(nn.Linear(hdim, odim, bias=False))
        
    def forward(self, x):
        '''BLSTMP forward
        :param xs:
        :param ilens:
        :return:
        '''
        x = self.rnns(x)
        x = self.fc(x)
        return x  # x: utt list of frame x dim
        