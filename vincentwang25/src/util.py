import numpy as np
import torch
import audiomentations as A
from torch.utils.data import Dataset

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
    
