import os
import time
import shutil
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel

from data.test_data_loader import DeepSpeakerTestDataset, DeepSpeakerTestDataLoader
from data.seq_data_loader import DeepSpeakerSeqDataset, DeepSpeakerSeqDataLoader
from model.model import model_select, load
from train.base_trainer import reduce_loss_dict, evaluate
from utils import utils


def train(opt, logging):    
    ## Data Prepare ##
    if opt.main_proc:
        logging.info("Building dataset")
                                           
    train_dataset = DeepSpeakerSeqDataset(opt, os.path.join(opt.dataroot, 'dev'))
    train_loader = DeepSpeakerSeqDataLoader(train_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=True, pin_memory=True)
             
    val_dataset = DeepSpeakerTestDataset(opt, os.path.join(opt.dataroot, 'test'))
    val_loader = DeepSpeakerTestDataLoader(val_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=False, pin_memory=True)
    
    opt.in_size = train_dataset.in_size
    opt.out_size = train_dataset.class_nums  
    print('opt.in_size {} opt.out_size {}'.format(opt.in_size, opt.out_size))  
                                           
    if opt.main_proc:
        logging.info("Building dataset Sucessed")
    
    ##  Building Model ##
    if opt.main_proc:
        logging.info("Building Model")
    
    model = model_select(opt, seq_training=True)
    
    if opt.resume:
        model, opt.total_iters = load(model, opt.resume, 'state_dict')
    
    # define optimizers for different layer
    if opt.optim_type == 'sgd':
        optimizer = optim.SGD([
            {'params': model.parameters(), 'weight_decay': 5e-4},
        ], lr=opt.lr, momentum=0.9, nesterov=True)
    elif opt.optim_type == 'adam':
        optimizer = optim.Adam([
            {'params': model.parameters(), 'weight_decay': 5e-4},
        ], lr=opt.lr, betas=(opt.beta1, 0.999))
        
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=opt.lr_reduce_step, gamma=opt.lr_reduce_factor, last_epoch=-1)
        
    model.to(opt.device)
    
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                          output_device=opt.local_rank)
    if opt.main_proc:
        print(model)
        logging.info("Building Model Sucessed") 
        
    best_perform_acc = 1.0
    
    losses = utils.AverageMeter()
    embedding_losses = utils.AverageMeter()
    embedding_segment_losses = utils.AverageMeter()
    penalty_losses = utils.AverageMeter()
    embedding_segment_losses = utils.AverageMeter()

    # Initial performance
    if opt.main_proc:
        EER = evaluate(opt, model, val_loader, logging)
        best_perform_acc = EER
        print('>>Start performance: EER = {}<<'.format(best_perform_acc))
    
    save_model = model
    if isinstance(model, DistributedDataParallel):
        save_model = model.module
    
    # Start Training
    total_iters = opt.total_iters
    for epoch in range(1, opt.total_epoch + 1):
        while True:
            model.train()
            for i, (data) in enumerate(train_loader, start=0):
                if i == len(train_loader):
                    break

                optimizer.zero_grad()

                # Perform forward and Obtain the loss
                feature_input, seq_len, spk_ids = data              
                feature_input = feature_input.to(opt.device)
                seq_len = seq_len.squeeze(0).to(opt.device)
                out, out_segment, attn, w, b = model(feature_input, seq_len) 
                
                sim_matrix_out = save_model.similarity(out, w, b)
                embedding_loss = opt.embedding_loss_lamda * save_model.loss_cal(sim_matrix_out)
        
                if opt.segment_type == 'average':
                    sim_matrix_out_seg = save_model.similarity(out_segment, w, b)
                    embedding_loss_segment = opt.segment_loss_lamda * save_model.loss_cal(sim_matrix_out_seg)
                elif opt.segment_type == 'all':
                    sim_matrix_out_seg = save_model.similarity_segment(out_segment, seq_len, w, b)
                    embedding_loss_segment = opt.segment_loss_lamda * save_model.loss_cal_segment(sim_matrix_out_seg, seq_len)
                else:
                    sim_matrix_out_seg = None
                    embedding_loss_segment = 0
            
                if opt.att_type == 'multi_attention' and attn is not None:
                    penalty_loss = opt.penalty_loss_lamda * save_model.penalty_loss_cal(attn)
                else:
                    penalty_loss = 0
            
                loss_dict_reduced = reduce_loss_dict(opt, {'embedding_loss': embedding_loss, 'penalty_loss': penalty_loss, 
                                                           'embedding_loss_segment': embedding_loss_segment})                
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()
                embedding_loss_value = loss_dict_reduced['embedding_loss'].item()
                penalty_loss_value = loss_dict_reduced['penalty_loss'].item()
                embedding_loss_segment_value = loss_dict_reduced['embedding_loss_segment'].item()
                loss = embedding_loss + penalty_loss

                # Check the loss and avoid the invaided loss
                inf = float("inf")
                if loss_value == inf or loss_value == -inf:
                    print("WARNING: received an inf loss, setting loss value to 0")
                    loss_value = 0
                    embedding_loss_value = 0
                    penalty_loss_value = 0
                    embedding_loss_segment_value = 0
                    continue

                # Perform backward and Check and update the grad
                loss.backward()
                if utils.check_grad(model.parameters(), opt.clip_grad, opt.ignore_grad):
                    if opt.main_proc:
                        logging.info('Not a finite gradient or too big, ignoring')
                    optimizer.zero_grad()
                    continue
                optimizer.step()
    
                total_iters += opt.num_gpus

                # Update the loss for logging
                losses.update(loss_value)
                embedding_losses.update(embedding_loss_value)
                penalty_losses.update(penalty_loss_value)
                embedding_segment_losses.update(embedding_loss_segment_value)

                # Print the performance on the training dateset 'opt': opt, 'learning_rate': lr,
                if total_iters % opt.print_freq == 0:
                    scheduler.step(total_iters)
                    if opt.main_proc:
                        lr = scheduler.get_lr()
                        if isinstance(lr, list):
                            lr = max(lr)
                        logging.info(
                            '==> Train set steps {} lr: {:.6f}, loss: {:.4f} [ embedding: {:.4f}, embedding_segment: {:.4f}, penalty_loss {:.4f}]'.format(
                                total_iters, lr, losses.avg, embedding_losses.avg, embedding_segment_losses.avg, penalty_losses.avg))
        
                        save_model = model
                        if isinstance(model, DistributedDataParallel):
                            save_model = model.module
                        state = {'state_dict': save_model.state_dict(), 'total_iters': total_iters}
                        filename = 'newest_model.pth'
                        if os.path.isfile(os.path.join(opt.model_dir, filename)):
                            shutil.copy(os.path.join(opt.model_dir, filename), os.path.join(opt.model_dir, 'newest_model.pth_bak'))
                        utils.save_checkpoint(state, opt.model_dir, filename=filename)

                # Validate the trained model
                if total_iters % opt.validate_freq == 0:
                    EER = evaluate(opt, model, val_loader, logging)
                    ##scheduler.step(EER)
                    
                    if opt.main_proc and EER < best_perform_acc:
                        best_perform_acc = EER
                        print("Found better validated model (EER = %.3f), saving to model_best.pth" % (best_perform_acc))
                        save_model = model
                        if isinstance(model, DistributedDataParallel):
                            save_model = model.module
                        state = {'state_dict': save_model.state_dict(), 'total_iters': total_iters}
                        filename = 'model_best.pth'
                        if os.path.isfile(os.path.join(opt.model_dir, filename)):
                            shutil.copy(os.path.join(opt.model_dir, filename), os.path.join(opt.model_dir, 'model_best.pth_bak'))
                        utils.save_checkpoint(state, opt.model_dir, filename=filename)                             
    
                    model.train()
                    losses.reset()
                    embedding_losses.reset()
                    penalty_losses.reset()
                    embedding_segment_losses.reset()
    
                if total_iters > opt.max_iters and opt.main_proc:
                    logging.info('finish training, steps is  {}'.format(total_iters))
                    return model
