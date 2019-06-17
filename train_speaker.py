from __future__ import print_function

import os
import random
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

from options.config import TrainOptions
from utils.utils import create_output_dir

# Prepare the parameters
opt = TrainOptions().parse()

## set seed ##
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print('manual_seed = %d' % opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

# Configure the distributed training
opt.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
opt.distributed = opt.num_gpus > 1
if opt.cuda:
    opt.device = torch.device("cuda")
else:
    opt.device = torch.device("cpu")

opt.main_proc = True
if opt.distributed:
    print("gpu_ids = %s" % (opt.local_rank))
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url, world_size=opt.num_gpus, rank=opt.local_rank)
    opt.main_proc = opt.local_rank == 0

## prepara dir for training ##
if not os.path.isdir(opt.works_dir):
    try:
        os.makedirs(opt.works_dir)
    except OSError:
        print("ERROR: %s is not a dir" % (opt.works_dir))
        pass

opt.exp_path = os.path.join(opt.works_dir, 'exp')
if not os.path.isdir(opt.exp_path):
    os.makedirs(opt.exp_path)

opt.model_name = 'speaker_{}-{}-{}-{}'.format(opt.model_type, opt.loss_type, opt.att_type, opt.margin_type)
opt.log_dir = os.path.join(opt.exp_path, opt.model_name)
if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

opt.model_dir = os.path.join(opt.exp_path, opt.model_name)
if not os.path.exists(opt.model_dir):
    os.makedirs(opt.model_dir)

if opt.cmvn_file is not None:
    opt.cmvn_file = os.path.join(opt.model_dir, opt.cmvn_file)
    
## Create a logger for loging the training phase ##
logging = create_output_dir(opt, opt.log_dir)
    
if opt.loss_type.split('_')[0] == 'ge2e':
    from train.ge2e_trainer import train
elif opt.loss_type.split('_')[0] == 'seq':
    from train.seq_trainer import train                                      
elif opt.loss_type.split('_')[0] == 'class':
    from train.class_trainer import train

train(opt, logging)
