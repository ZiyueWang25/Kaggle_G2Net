class Config:

    #frequently changed 
    model_version = "120th_V2_PL_6ep_1em3lr_32ch_vf_s01" 
    model_module = 'ModelIafossV2'
    use_pseudo_label = True
    
    #conservative
    conservative_aug = []
    #aggressive, OneOf 
    aggressive_aug_proba = 0.75
    aggressive_aug = ['vflip','add_gaussian_noise','shuffle01','timemask','time_shift',]     #'reduce_SNR'
    
    
    vflip = True
    vflip_weight = 1.0 
    add_gaussian_noise = False 
    add_gaussian_noise_weight = 1.0    
    timemask = False
    timemask_weight = 1.0
    shuffle01 = True
    shuffle01_weight = 1.0
    time_shift = False
    time_shift_left = 96
    time_shift_right = 96
    time_shift_weight = 0.5
    
    
    pseudo_label_folder = output_dir + "main_112th_V2SD_PL_6ep_5Fold/"#main_35th_GeM_vflip_shuffle01_5fold,#main_112th_V2SD_PL_6ep_5Fold

    epochs = 6
    batch_size = 64
    lr=  1e-3
    
    #CNN structure
    channels = 32
