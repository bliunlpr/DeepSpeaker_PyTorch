import os
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils import distributed_util
from utils import utils


def reduce_loss_dict(opt, loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = distributed_util.get_world_size()
    if world_size < 2:
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            if not torch.is_tensor(v):
                v = torch.tensor(0.0).to(opt.device)
            all_losses.append(v)
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
        return reduced_losses
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            if not torch.is_tensor(v):
                v = torch.tensor(0.0).to(opt.device)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def extract_feature(opt, inputs, model, segment_length, segment_shift):
    input_data = None
    length = inputs.size(1)
    seq_len = 0
    for x in range(0, length, segment_shift):
        end = x + segment_length
        if end < length:
            feature_mat = inputs[:, x:end, :]
        else:
            if x == 0:
                input_data = inputs
                seq_len += 1
            break
        seq_len += 1
        if input_data is None:
            input_data = feature_mat
        else:
            input_data = torch.cat((input_data, feature_mat), 0) 
    seq_len = torch.LongTensor([seq_len]).to(opt.device) 
    output, w, b = model(input_data, seq_len)
    output_mean = torch.mean(output, dim=0)
    output_mean_norm = output_mean / torch.sqrt(torch.sum(output_mean**2, dim=-1, keepdim=True) + 1e-6).unsqueeze(0)      
    return output_mean_norm
    
    
def evaluate(opt, model, val_loader, logging):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    model.eval()

    # show progress bar only on main process.
    valid_enum = tqdm(val_loader, desc='Valid') if distributed_util.is_main_process() else iter(val_loader)

    segment_length = int((opt.min_segment_length + opt.max_segment_length) / 2)
    segment_shift = int(opt.segment_shift_rate * segment_length)
    probs = []
    labels = []

    torch.set_grad_enabled(False)
    for i, (data) in enumerate(valid_enum, start=0):
        with torch.no_grad():                
            utt_id_list, enroll_feat_list, test_feat, label = data
            enroll_output_all = None
            for enroll_feat in enroll_feat_list:
                inputs = enroll_feat.to(opt.device)
                enroll_output = extract_feature(opt, inputs, model, segment_length, segment_shift)
                if enroll_output_all is None:
                    enroll_output_all = enroll_output
                else:
                    enroll_output_all = torch.cat((enroll_output_all, enroll_output), dim=0)  
            enroll_output_mean = torch.mean(enroll_output_all, dim=0)
            enroll_output_mean_norm = enroll_output_mean / torch.sqrt(torch.sum(enroll_output_mean**2, dim=-1, keepdim=True) + 1e-6)
            enroll_output_mean_norm = enroll_output_mean_norm.detach().cpu().numpy()     
                   
            inputs = test_feat.to(opt.device)
            test_output = extract_feature(opt, inputs, model, segment_length, segment_shift)
            test_output = test_output.detach().cpu().numpy()     
            
            prob = float(np.sum(enroll_output_mean_norm * test_output))
            probs.append(prob)
            labels.append(int(label.detach().cpu().numpy()))
                    
    model.train()
    torch.set_grad_enabled(True)
    
    eer, thresh = utils.processDataTable2(np.array(labels), np.array(probs))
    if opt.main_proc:
        logging.info("EER : %0.4f (thres:%0.4f)" % (eer, thresh))
    return eer
