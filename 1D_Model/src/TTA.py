import audiomentations as A
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, PolarityInversion
from torch.utils.data import Dataset

import torch
import numpy as np


class TTA(Dataset):
    def __init__(self, paths, targets, use_vflip=False, shuffle_channels=False, time_shift=False,
                 add_gaussian_noise=False, time_stretch=False, shuffle01=False):
        self.paths = paths
        self.targets = targets
        self.use_vflip = use_vflip
        self.shuffle_channels = shuffle_channels
        self.time_shift = time_shift
        self.gaussian_noise = add_gaussian_noise
        self.time_stretch = time_stretch
        self.shuffle01 = shuffle01
        if time_shift:
            self.time_shift = A.Shift(min_fraction=-512 * 1.0 / 4096, max_fraction=-1.0 / 4096, p=1, rollover=False)
        if add_gaussian_noise:
            self.gaussian_noise = A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1)
        if time_stretch:
            self.time_stretch = A.TimeStretch(min_rate=0.9, max_rate=1.111, leave_length_unchanged=True, p=1)

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
            waves[[0, 1]] = waves[[1, 0]]

        waves = torch.FloatTensor(waves * 1e20)
        target = torch.tensor(self.targets[index], dtype=torch.float)  # device=device,
        return (waves, target)


def addAugmentation(Config, name, cons_func, agg_fun, cons_list, cons_name_list, agg_list, agg_name_list):
    if name in Config.aggressive_aug:
        agg_list.append(agg_fun)
        agg_name_list.append(name)
    else:
        cons_list.append(cons_func)
        cons_name_list.append(name)


def get_tranform_list(Config):

    cons_funcs = []
    aggr_funcs = []
    cons_func_names = []
    aggr_funcs_names = []

    if Config.vflip:
        def vflip_func(x, p):
            return -x if np.random.random() < p else x

        addAugmentation(Config, 'vflip',
                        lambda x: vflip_func(x, Config.vflip_proba),
                        lambda x: vflip_func(x, 1),
                        cons_funcs, cons_func_names,
                        aggr_funcs, aggr_funcs_names)

    if Config.add_gaussian_noise:
        addAugmentation(Config, 'add_gaussian_noise',
                        A.AddGaussianNoise(min_amplitude=0.001 * 0.015, max_amplitude=0.015 * 0.015, p=Config.add_gaussian_noise_proba),
                        A.AddGaussianNoise(min_amplitude=0.001 * 0.015, max_amplitude=0.015 * 0.015, p=1),
                        cons_funcs, cons_func_names,
                        aggr_funcs, aggr_funcs_names)

    if Config.timemask:
        addAugmentation(Config, 'timemask',
                        A.TimeMask(min_band_part=0.0, max_band_part=0.03, fade=False, p=Config.timemask_proba),
                        A.TimeMask(min_band_part=0.0, max_band_part=0.03, fade=False, p=1),
                        cons_funcs, cons_func_names,
                        aggr_funcs, aggr_funcs_names)

    if Config.shuffle01:
        def shuffle01_func(x, p=1):
            return x[[1, 0, 2]] if np.random.random() < p else x
        addAugmentation(Config, 'shuffle01',
                        lambda x: shuffle01_func(x, Config.shuffle01_proba),
                        lambda x: shuffle01_func(x, 1),
                        cons_funcs, cons_func_names,
                        aggr_funcs, aggr_funcs_names)


    if Config.time_shift:
        def time_shift(p):
            return A.Shift(min_fraction=-Config.time_shift_left * 1.0 / 4096,
                        max_fraction=Config.time_shift_right * 1.0 / 4096,
                        p=p, rollover=False)  # <0 means shift towards left,  fraction of total sound length
        addAugmentation(Config, 'time_shift',
                        time_shift(Config.time_shift_proba),
                        time_shift(1),
                        cons_funcs, cons_func_names,
                        aggr_funcs, aggr_funcs_names)



    def shift_channel_func_random(x, sample_rate=2048):
        channel = np.random.choice(3)
        trans = A.Shift(min_fraction=-Config.shift_channel_left * 1.0 / 4096,
                        max_fraction=Config.shift_channel_right * 1.0 / 4096,
                        p=Config.shift_channel_proba, rollover=False)
        x[channel] = trans(x[channel], sample_rate=2048)
        return x

    if Config.shift_channel:
        def shift_channel(x, p):
            def shift_channel_func(x, sample_rate=2048):
                channel = np.random.choice(3)
                trans = A.Shift(min_fraction=-Config.shift_channel_left * 1.0 / 4096,
                                max_fraction=Config.shift_channel_right * 1.0 / 4096,
                                p=p, rollover=False)
                x[channel] = trans(x[channel], sample_rate=2048)
                return x
            return shift_channel_func

        addAugmentation(Config, 'shift_channel',
                        lambda x: shift_channel(x, Config.shift_channel_proba),
                        lambda x: shift_channel(x, 1),
                        cons_funcs, cons_func_names,
                        aggr_funcs, aggr_funcs_names)

    if Config.reduce_SNR:
        def reduce_SNR_func(x, sample_rate=2048, p):
            multiplier = math.sqrt(1 - Config.reduce_SNR_ratio ** 2)
            x = x * Config.reduce_SNR_ratio
            trans = A.AddGaussianNoise(min_amplitude=multiplier, max_amplitude=multiplier, p=1)
            x = trans(x, sample_rate=2048)
            return x

        addAugmentation(Config, 'reduce_SNR',
                        lambda x: reduce_SNR_func(x, p=Config.reduce_SNR_proba),
                        lambda x: reduce_SNR_func(x, p=1),
                        cons_funcs, cons_func_names,
                        aggr_funcs, aggr_funcs_names)

    print('conservative transforms: ', cons_func_names)
    print('aggressive transforms: ', aggr_func_names)
    train_transform = cons_funcs  # A.Compose(conserv_transform_list)#,OneOf(aggressive_transform_list,p=0.5)) # no OneOf in audiomentation
    test_transform = None  # A.Compose([])
    return train_transform, test_transform