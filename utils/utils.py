import numpy as np
import time
import torch
import os
import sys
import logging
import time
from datetime import timedelta
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)
        
def create_output_dir(opt, log_path):
    logfile = os.path.join(log_path, 'logger')

    # Safety check
    if os.path.exists(logfile) and opt.resume == "":
        logging.warning("Experiment already exists!")

    # Create logger
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(logfile, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # quite down visdom
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info(opt)
    return logger

def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm_(params, clip_th)
    return (not np.isfinite(befgad) or (befgad > ignore_th))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_acc(output, target):
    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    batch_size = target.size(0)
    correct *= (100.0 / batch_size)
    return correct         

def save_checkpoint(state, save_path, is_best=False, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename),
                        os.path.join(save_path, 'model_best.pth.tar'))

def adjust_learning_rate_by_factor(optimizer, lr, factor):
    """Adjusts the learning rate according to the given factor"""
    lr = lr * factor
    lr = max(lr, 0.000005)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
                                        
def processDataTable2(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh
