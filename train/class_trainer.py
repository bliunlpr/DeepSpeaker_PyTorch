import os
import time
import shutil
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel

from data.test_data_loader import DeepSpeakerTestDataset, DeepSpeakerTestDataLoader
from data.utt_data_loader import DeepSpeakerUttDataset, DeepSpeakerUttDataLoader
from data.data_sampler import BucketingSampler, DistributedBucketingSampler
from model.model import model_select, load
from model.margin import margin_select
from train.base_trainer import reduce_loss_dict, evaluate
from utils import utils


def train(opt, logging):
    
    ## Data Prepare ##
    if opt.main_proc:
        logging.info("Building dataset")
                                           
    train_dataset = DeepSpeakerUttDataset(opt, os.path.join(opt.dataroot, 'dev'))
    if not opt.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=opt.batch_size,
                                                    num_replicas=opt.num_gpus, rank=opt.local_rank)
    train_loader = DeepSpeakerUttDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
             
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
    
    model = model_select(opt)
    margin = margin_select(opt)
    
    if opt.resume:
        model, opt.total_iters = load(model, opt.resume, 'state_dict')
        margin, opt.total_iters = load(margin, opt.resume, 'margin_state_dict')
    
    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    if opt.optim_type == 'sgd':
        optimizer = optim.SGD([
            {'params': model.parameters(), 'weight_decay': 5e-4},
            {'params': margin.parameters(), 'weight_decay': 5e-4}
        ], lr=opt.lr, momentum=0.9, nesterov=True)
    elif opt.optim_type == 'adam':
        optimizer = optim.Adam([
            {'params': model.parameters(), 'weight_decay': 5e-4},
            {'params': margin.parameters(), 'weight_decay': 5e-4}
        ], lr=opt.lr, betas=(opt.beta1, 0.999))
        
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[8, 14, 20], gamma=0.1)
        
    model.to(opt.device)
    margin.to(opt.device)
    
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                          output_device=opt.local_rank)
        margin = torch.nn.parallel.DistributedDataParallel(margin, device_ids=[opt.local_rank],
                                                           output_device=opt.local_rank)
    if opt.main_proc:
        print(model)
        print(margin)
        logging.info("Building Model Sucessed") 
        
    best_perform_eer = 1.0
    
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    # Initial performance
    if opt.main_proc:
        EER = evaluate(opt, model, val_loader, logging)
        best_perform_eer = EER
        print('>>Start performance: EER = {}<<'.format(best_perform_eer))
    
    total_iters = opt.total_iters
    for epoch in range(1, opt.total_epoch + 1):
        train_sampler.shuffle(epoch)
        scheduler.step()
        # train model
        if opt.main_proc:
            logging.info('Train Epoch: {}/{} ...'.format(epoch, opt.total_epoch))
        model.train()
        margin.train()

        since = time.time()
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, inputs, targets = data
            inputs, label = inputs.to(opt.device), targets.to(opt.device)
            optimizer.zero_grad()
            
            raw_logits, attn, w, b = model(inputs)
            output = margin(raw_logits, label)
            loss = criterion(output, label)
            loss_dict_reduced = reduce_loss_dict(opt, {'loss': loss})
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            
            # Check the loss and avoid the invaided loss
            inf = float("inf")
            if loss_value == inf or loss_value == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
                continue
                    
            loss.backward()
            if utils.check_grad(model.parameters(), opt.clip_grad, opt.ignore_grad):
                if opt.main_proc:
                    logging.info('Not a finite gradient or too big, ignoring')
                optimizer.zero_grad()
                continue
            optimizer.step()

            total_iters += opt.num_gpus
            losses.update(loss_value)
            
            # print train information
            if total_iters % opt.print_freq == 0 and opt.main_proc:
                # current training accuracy
                _, predict = torch.max(output.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                logging.info("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f} ({:.4f}), train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(total_iters, epoch, loss_value, losses.avg, correct/total, time_cur, scheduler.get_lr()[0]))
              
            # save model
            if total_iters % opt.save_freq == 0 and opt.main_proc:
                logging.info('Saving checkpoint: {}'.format(total_iters))
                if opt.distributed:
                    model_state_dict = model.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                    margin_state_dict = margin.state_dict()
                state = {'state_dict': model_state_dict, 'margin_state_dict': margin_state_dict, 'total_iters': total_iters,}
                filename = 'newest_model.pth'
                if os.path.isfile(os.path.join(opt.model_dir, filename)):
                    shutil.copy(os.path.join(opt.model_dir, filename), os.path.join(opt.model_dir, 'newest_model.pth_bak'))
                utils.save_checkpoint(state, opt.model_dir, filename=filename)
                    
            # Validate the trained model
            if total_iters % opt.validate_freq == 0:
                EER = evaluate(opt, model, val_loader, logging)
                ##scheduler.step(EER)
                
                if opt.main_proc and EER < best_perform_eer:
                    best_perform_acc = EER
                    logging.info("Found better validated model (EER = %.3f), saving to model_best.pth" % (best_perform_acc))
                    if opt.distributed:
                        model_state_dict = model.module.state_dict()
                        margin_state_dict = margin.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()
                        margin_state_dict = margin.state_dict()
                    state = {'state_dict': modelt_state_dict, 'margin_state_dict': margin_state_dict, 'total_iters': total_iters,}  
                    filename = 'model_best.pth'
                    if os.path.isfile(os.path.join(opt.model_dir, filename)):
                        shutil.copy(os.path.join(opt.model_dir, filename), os.path.join(opt.model_dir, 'model_best.pth_bak'))                   
                    utils.save_checkpoint(state, opt.model_dir, filename=filename)

                model.train()
                margin.train()
                losses.reset()
                   