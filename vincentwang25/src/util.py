import random
import os
import numpy as np
import torch
import audiomentations as A
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset
from torch.optim.swa_utils import update_bn, AveragedModel, SWALR
from numba import jit

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


class TTA(Dataset):
    def __init__(self, paths, targets, use_vflip=False, shuffle_channels=False, time_shift=False, add_gaussian_noise = False,  time_stretch=False,shuffle01=False ):
        self.paths = paths
        self.targets = targets
        self.use_vflip = use_vflip
        self.shuffle_channels = shuffle_channels
        self.time_shift = time_shift
        self.gaussian_noise = add_gaussian_noise
        self.time_stretch = time_stretch
        self.shuffle01 = shuffle01
        if time_shift:
            self.time_shift = A.Shift(min_fraction=-512*1.0/4096, max_fraction=-1.0/4096, p=1,rollover=False)
        if add_gaussian_noise:
            self.gaussian_noise = A.AddGaussianNoise(min_amplitude=0.001, max_amplitude= 0.015, p=1)
        if time_stretch:
            self.time_stretch = A.TimeStretch(min_rate=0.9, max_rate=1.111,leave_length_unchanged=True, p=1)
              
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index] 
        waves = np.load(path)

        if self.use_vflip:
            waves = -waves
        if self.shuffle_channels:
            np.random.shuffle(waves)
        if self.time_shift:
            waves = self.time_shift(waves, sample_rate=2048)
        if self.gaussian_noise:
            waves = self.gaussian_noise(waves, sample_rate=2048)
        if self.time_stretch:
            waves = self.time_stretch(waves, sample_rate=2048)
        if self.shuffle01:
            waves[[0,1]] = waves[[1,0]]
        
        waves = torch.FloatTensor(waves * 1e20)
        target = torch.tensor(self.targets[index],dtype=torch.float)#device=device,             
        return (waves, target)
    

def get_swa(model, optimizer, epochs, swa_start_step_epoch, swa_lr, train_size, batch_size):
    swa_model = AveragedModel(model)
    epoch_step = train_size/batch_size        
    swa_start_step =  epoch_step * swa_start_step_epoch
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
                                                anneal_strategy="cos", 
                                                anneal_epochs=int(epoch_step * (epochs-1)), 
                                                swa_lr=swa_lr)
    return swa_model, swa_scheduler

def do_swa_scheduler(step, swa_scheduler, swa_start_step):
    if (swa_scheduler is not None) and (step >= swa_start_step):
        return True
    else:
        return False

def get_score(y_true, y_pred):
    score = fast_auc(y_true, y_pred)
    return score

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_scheduler(optimizer, train_size, batch_size, epochs, warmup=0.1):
    epoch_step = train_size/batch_size
    num_warmup_steps = int(warmup * epoch_step * epochs)
    num_training_steps = int(epoch_step * epochs)
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

def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

class Dict2Class():
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
            
def id_2_path_wave(file_id: str, input_dir=".", train=True) -> str:
    if train:
        return "{}/train-float32/{}.npy".format(input_dir, file_id)
    else:
        return "{}/test-float32/{}.npy".format(input_dir, file_id)

    
def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Number of device:', torch.cuda.device_count())
    return device

def get_key(path):
    f = open(path, "r")
    key = f.read().strip()
    f.close()
    return key
