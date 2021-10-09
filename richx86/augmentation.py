import math
import numpy as np
from config import Config
import audiomentations as A
def augmentations():
    conserv_transform_list = []
    aggressive_transform_list = []
    conserv_transform_list_strings = []
    aggressive_transform_list_strings = []
    
    #-------------------------vflip
    if Config.vflip:
    #     trans = lambda x:-x
        def vflip_func(x,sample_rate=2048):
            return -x
        def vflip_func_random(x,sample_rate=2048):
            if np.random.random()<Config.vflip_proba:
                return -x
            else:
                return x
        if 'vflip' in Config.aggressive_aug:
            aggressive_transform_list.append(vflip_func)
            aggressive_transform_list_strings.append('vflip')
        else:
            conserv_transform_list.append(vflip_func_random)
            conserv_transform_list_strings.append('vflip')
    #----------------------add_gaussian_noise        
    if Config.add_gaussian_noise:
        
        if 'add_gaussian_noise' in Config.aggressive_aug:
            trans = A.AddGaussianNoise(min_amplitude=0.001*0.015, max_amplitude=0.015*0.015, p=1) #tbs #0.015 is the estimated std
            aggressive_transform_list.append(trans)
            aggressive_transform_list_strings.append('add_gaussian_noise')
        else:
            trans = A.AddGaussianNoise(min_amplitude=0.001*0.015, max_amplitude=0.015*0.015, p=Config.add_gaussian_noise_proba) #tbs #0.015 is the estimated std
            conserv_transform_list.append(trans)
            conserv_transform_list_strings.append('add_gaussian_noise')
    
    #--------------------------timemask
    if Config.timemask:
        
        if 'timemask' in Config.aggressive_aug:
            trans = A.TimeMask(min_band_part=0.0, max_band_part=0.03, fade=False, p=1)
            aggressive_transform_list.append(trans)
            aggressive_transform_list_strings.append('timemask')
        else:
            trans = A.TimeMask(min_band_part=0.0, max_band_part=0.03, fade=False, p=Config.timemask_proba)
            conserv_transform_list.append(trans)
            conserv_transform_list_strings.append('timemask')
    
    #--------------------------shuffle01        
    def shuffle01_func(x,sample_rate=2048):
        return x[[1,0,2]]
    def shuffle01_func_random(x,sample_rate=2048):
        if np.random.random()<Config.shuffle01_proba: 
            return x[[1,0,2]]
        else:
            return x
    if Config.shuffle01:
    #     trans = lambda x:x[[1,0,2]]
    
        if 'shuffle01' in Config.aggressive_aug:
            aggressive_transform_list.append(shuffle01_func)
            aggressive_transform_list_strings.append('shuffle01')
        else:
            conserv_transform_list.append(shuffle01_func_random)
            conserv_transform_list_strings.append('shuffle01')
    #---------------------------time_shift
    if Config.time_shift:
        if 'time_shift' in Config.aggressive_aug:
            trans = A.Shift(min_fraction=-Config.time_shift_left*1.0/4096,
                            max_fraction=Config.time_shift_right*1.0/4096, 
                            p=1,rollover=False)#<0 means shift towards left,  fraction of total sound length
            aggressive_transform_list.append(trans)
            aggressive_transform_list_strings.append('time_shift')
        else:
            trans = A.Shift(min_fraction=-Config.time_shift_left*1.0/4096,
                                    max_fraction=Config.time_shift_right*1.0/4096, 
                                    p=Config.time_shift_proba,rollover=False)
            conserv_transform_list.append(trans)
            conserv_transform_list_strings.append('time_shift')
    
    #-----------------shift_channel        
    def shift_channel_func(x,sample_rate=2048):
        channel = np.random.choice(3)
        trans = A.Shift(min_fraction=-Config.shift_channel_left*1.0/4096,
                    max_fraction=Config.shift_channel_right*1.0/4096, 
                    p=1,rollover=False)
        x[channel] = trans(x[channel],sample_rate=2048)
        return x
    def shift_channel_func_random(x,sample_rate=2048):
        channel = np.random.choice(3)
        trans = A.Shift(min_fraction=-Config.shift_channel_left*1.0/4096,
                    max_fraction=Config.shift_channel_right*1.0/4096, 
                    p=Config.shift_channel_proba,rollover=False)
        x[channel] = trans(x[channel],sample_rate=2048)
        return x
    if Config.shift_channel:
        if 'shift_channel' in Config.aggressive_aug:
            
            aggressive_transform_list.append(shift_channel_func)
            aggressive_transform_list_strings.append('shift_channel')
        else:
            
            conserv_transform_list.append(shift_channel_func_random)
            conserv_transform_list_strings.append('shift_channel')
    #-----------------reduce_SNR        
    def reduce_SNR_func(x,sample_rate=2048):
        x = x * Config.reduce_SNR_ratio
        trans = A.AddGaussianNoise(min_amplitude=multiplier, max_amplitude=multiplier, p=1)
        x = trans(x,sample_rate=2048)
        return x 
    def reduce_SNR_func_random(x,sample_rate=2048):
        if np.random.random() < Config.reduce_SNR_proba:
            x = x * Config.reduce_SNR_ratio
            trans = A.AddGaussianNoise(min_amplitude=multiplier, max_amplitude=multiplier, p=1)
            x = trans(x,sample_rate=2048)
        return x
    if Config.reduce_SNR:
        multiplier = math.sqrt(1-Config.reduce_SNR_ratio**2)
        if 'reduce_SNR' in Config.aggressive_aug:
    
            aggressive_transform_list.append(reduce_SNR_func)
            aggressive_transform_list_strings.append('reduce_SNR')
        else:
    
            conserv_transform_list.append(reduce_SNR_func_random)
            conserv_transform_list_strings.append('reduce_SNR')
            
    
    print('conservative transforms: ',conserv_transform_list_strings)
    print('aggressive transforms: ',aggressive_transform_list_strings)
    
    
    return conserv_transform_list, aggressive_transform_list, conserv_transform_list_strings, aggressive_transform_list_strings

