import argparse
import os
import utils
import torch


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--works_dir', help='path to work', default='.')
        self.parser.add_argument('--dataroot', required=True, help='path to feats.scp utt2spk utt2data utt2vad (should have subfolders train, dev, test)')
        self.parser.add_argument('--model_name', type=str, default='speaker', help='name of the experiment.')
        
        self.parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use gpu to train')
        self.parser.add_argument("--local_rank", type=int, default=0)
        self.parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
        self.parser.add_argument('--dist-backend', default='nccl', type=str, help='nccl gloo distributed backend')
        
        self.parser.add_argument('--vad', dest='vad', action='store_true', help='Use vad to remove the sil')
        self.parser.add_argument('--delta_order', type=int, default=0, help='input delta-order')
        self.parser.add_argument('--normalize_type', type=int, default=1, help='normalize_type')
        self.parser.add_argument('--cmvn_file', default=None, type = str, help='File to cmvn')
        self.parser.add_argument('--range_norm_file', default=None, type = str, help='File to range_norm')
        self.parser.add_argument('--num_utt_cmvn', type=int, default=20000, help='the number of utterances for cmvn')
        self.parser.add_argument('--left_context_width', type=int, default=0, help='input left_context_width-width')
        self.parser.add_argument('--right_context_width', type=int, default=0, help='input right_context_width')
        
        self.parser.add_argument('--segment_shift_rate', type=float, default=0.75, help='frame_slice_steps')   
        self.parser.add_argument('--min_segment_length', type=int, default=90, help='how many batches to print the trained model') 
        self.parser.add_argument('--max_segment_length', type=int, default=160, help='how many batches to print the trained model')
        self.parser.add_argument('--min_num_segment', type=int, default=5, help='min number of segment cutted from a utt') 
        self.parser.add_argument('--max_num_segment', type=int, default=25, help='max number of segment cutted from a utt')
    
        self.parser.add_argument('--num_workers', default=6, type=int, help='Number of workers used in data-loading')
        self.parser.add_argument('--speaker_num', type=int, default=32, help='speaker_num')        
        self.parser.add_argument('--utter_num', type=int, default=10, help='utter_num')
        self.parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        
        self.parser.add_argument('--att_type', type=str, default='last_state', help='train_type, last_state|average_state|base_attention|multi_attention')
        self.parser.add_argument('--segment_type', type=str, default='none', help='train_type, average | all | none')
        self.parser.add_argument('--model_type', type=str, default='lstm', help='model_type, lstm|cnn')
        self.parser.add_argument('--rnn_hidden_size', type=int, default=128, help='rnn_hidden_size')
        self.parser.add_argument('--embedding_size', type=int, default=128, help='embedding_size')       
        self.parser.add_argument('--rnn_layers', type=int, default=3, help='dnn_num_layer')
        self.parser.add_argument('--dropout', type=float, default=0.0, help='ignore grad before clipping')
        self.parser.add_argument('--rnn_type', default='lstm', help='Type of the functions. relu|sigmoid|tanh are supported')
        self.parser.add_argument('--rnn_bidirectional', default=False, action='store_true', help='bidirectional to dnn')
        self.parser.add_argument('--attention_dim', type=int, default=100, help='attention_dim')        
        self.parser.add_argument('--attention_head_num', type=int, default=3, help='attention_head_num')
        self.parser.add_argument('--embedding_loss_lamda', type=float, default=1.0, help='class_loss_lamda')
        self.parser.add_argument('--penalty_loss_lamda', type=float, default=0.01, help='penalty_loss_lamda') 
        self.parser.add_argument('--segment_loss_lamda', type=float, default=0.2, help='segment_loss_lamda')
        
        self.parser.add_argument('--margin_s', type=float, default=25.0, help='margin_s')
        self.parser.add_argument('--margin_m', type=float, default=0.5, help='margin_m')
        
        self.parser.add_argument('--checkpoints_dir', type=str, default='./exp', help='models are saved here')  
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')    

        self.parser.add_argument('--validate_freq', type=int, default=400, help='how many batches to validate the trained model')  
        self.parser.add_argument('--save_freq', type=int, default=200, help='how many batches to validate the save model')          
        self.parser.add_argument('--print_freq', type=int, default=200, help='how many batches to print the trained model') 
         
        self.parser.add_argument('--total_iters', default=0, type=int, metavar='N', help='manual hours number (useful on restarts)')
        self.parser.add_argument('--total_epoch', default=100, type=int, metavar='N', help='total_epoch')        
        self.parser.add_argument('--max_iters', default=120000, type=int, metavar='N', help='manual hours number (useful on restarts)')
        
        self.parser.add_argument('--optim_type', type=str, default='adam', help='optim_type adam|sgd')  
        self.parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0002')
        self.parser.add_argument('--min_lr', type=float, default=0.00001, help=' min learning rate, default=0.00001')
        self.parser.add_argument('--lr_reduce_factor', default=0.5, type=float, help='lr_reduce_factor')
        self.parser.add_argument('--step_patience', default=1, type=int, help='step_patience')
        self.parser.add_argument('--lr_reduce_threshold', default=1e-4, type=float, help='lr_reduce_threshold')
        self.parser.add_argument('--warmup_iters', default=8000, type=int, help='warmup_iters')
        self.parser.add_argument('--lr_reduce_step', default=20000, type=int, help='lr_reduce_step')

        self.parser.add_argument('--loss_type', type=str, default='class_softmax', help='train_type, softmax|contrast|margin')
        self.parser.add_argument('--margin_type', type=str, default='Softmax', help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')  
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
        self.parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
        self.parser.add_argument('--clip-grad', type=float, default=3.0, help='maximum norm of gradient clipping')
        self.parser.add_argument('--ignore-grad', type=float, default=100000.0, help='ignore grad before clipping')
        
        self.parser.add_argument('--manual_seed', type=int, help='manual seed', default = None)
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        return self.opt

