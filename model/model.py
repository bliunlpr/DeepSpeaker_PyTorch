import os
import torch
from model.rnn_model import DeepSpeakerRNNModel, DeepSpeakerRNNSeqModel
from model.cnn_model import DeepSpeakerCNNModel, DeepSpeakerCNNSeqModel


def model_load(checkpoint, state_dict, model):
    model_dict = model.state_dict()
    if state_dict in checkpoint:
        resume_dict = checkpoint[state_dict]
        resume_dict = {k: v for k, v in resume_dict.items() if k in model_dict}
        print ('resume_dict is {}'.format(resume_dict.keys()))
        model_dict.update(resume_dict)
        model.load_state_dict(model_dict)
    else:
        print('state_dict not in checkpoint')


def load(model, model_name, state_dict):
    total_iters = 0
    if os.path.isfile(model_name):
        checkpoint = torch.load(model_name)
        model_load(checkpoint, state_dict, model)
        total_iters = checkpoint['total_iters']
        print("=> loaded checkpoint '{}' (total_iters {})".format(model_name, total_iters))
    else:
        print("=> no checkpoint found at '{}'".format(model_name))
    return model, total_iters


def model_select(opt, seq_training=False):
    if seq_training:
        if 'lstm' in opt.model_type:
            model = DeepSpeakerRNNSeqModel(opt)
        elif 'cnn' in opt.model_type:
            model = DeepSpeakerCNNSeqModel(opt)
    else:
        if 'lstm' in opt.model_type:
            model = DeepSpeakerRNNModel(opt)
        elif 'cnn' in opt.model_type:
            model = DeepSpeakerCNNModel(opt)
    return model
    