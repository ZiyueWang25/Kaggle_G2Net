class Config:

    #frequently changed 

    model_version = "122nd_V2_PL_6ep_2em3lr_32ch_vf+gn+sc01+tm+ts" 
    model_module = 'ModelIafossV2'
    use_pseudo_label = True

    #conservative
    conservative_aug = []
    #aggressive, OneOf 
    aggressive_aug_proba = 0.80
    aggressive_aug = ['vflip','add_gaussian_noise','shuffle01','timemask','time_shift',]     #'reduce_SNR'
    

    vflip = True
    vflip_weight = 1.0 
    add_gaussian_noise = True 
    add_gaussian_noise_weight = 1.0    
    timemask = True
    timemask_weight = 0.8
    shuffle01 = True
    shuffle01_weight = 0.8
    time_shift = True
    time_shift_left = 96
    time_shift_right = 96
    time_shift_weight = 0.4
    

    pseudo_label_folder = output_dir+"main_112th_V2SD_PL_6ep_5Fold/"#main_35th_GeM_vflip_shuffle01_5fold,#main_112th_V2SD_PL_6ep_5Fold

    epochs = 6
    batch_size = 256
    lr=  2e-3 
  
    #CNN structure
    channels = 32
