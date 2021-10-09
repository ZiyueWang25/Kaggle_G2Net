import os
import random
import math
import numpy as np
import torch
# import torch_xla
# import torch_xla.core.xla_model as xm
from sklearn.metrics import roc_auc_score
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts,
                    CosineAnnealingLR, ReduceLROnPlateau,CyclicLR)
from config import Config

def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
  

def get_scheduler(optimizer, train_size):
    if Config.scheduler=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Config.factor, 
                                      patience=Config.patience, verbose=True, eps=Config.eps)
    elif Config.scheduler=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, 
                                      T_max=Config.T_max, 
                                      eta_min=Config.min_lr, last_epoch=-1)
    elif Config.scheduler=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                T_0=Config.T_0, 
                                                T_mult=1, 
                                                eta_min=Config.min_lr, 
                                                last_epoch=-1)
    elif Config.scheduler=='CyclicLR':
        iter_per_ep = train_size/Config.batch_size
        step_size_up = int(iter_per_ep*Config.step_up_epochs)
        step_size_down=int(iter_per_ep*Config.step_down_epochs)
        scheduler = CyclicLR(optimizer, 
                             base_lr=Config.base_lr, 
                             max_lr=Config.max_lr,
                             step_size_up=step_size_up,
                             step_size_down=step_size_down,
                             mode=Config.mode,
                             gamma=Config.cycle_decay**(1/(step_size_up+step_size_down)),
                             cycle_momentum=False)
        
    elif Config.scheduler == 'cosineWithWarmUp':
        epoch_step = train_size/Config.batch_size
        num_warmup_steps = int(0.1 * epoch_step * Config.epochs)
        num_training_steps = int(epoch_step * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=num_warmup_steps, 
                                                    num_training_steps=num_training_steps)      
    return scheduler
def mixed_criterion(loss_fn, pred, y_a, y_b, lam):
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, requires_grad=False).to(x.device,non_blocking=Config.non_blocking)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
def get_device():
    # setting device on GPU if available, else CPU
    if Config.use_tpu:
        
        device = xm.xla_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')#for debug, tb see
    print('Using device:', device)
    print()
    
    #Additional Info when using cuda
    # watch nvidia-smi
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Reserved:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    return device

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

