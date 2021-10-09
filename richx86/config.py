class Config:

    #frequently changed 
    model_name = 'TCNN'
    model_version = "test2" 
    model_module = 'Model1DCNNGEM'
    use_pseudo_label = False
 
    epochs = 12
    batch_size = 256
    lr= 5e-3
 
    # model_version = "133rd_V2SD_PL_4ep_2em3lr_32ch_vf_sc01_drop05" 
    # model_module = 'V2StochasticDepth'
    # use_pseudo_label = True
    
    output_dir = "G2Net-Model/"
    # pseudo_label_folder = output_dir+"120th_V2_PL_6ep_1em3lr_32ch_vf_s01/"
    
    

    

    
    
    debug = False
    use_checkpoint = False
    use_lr_finder = True
    use_subset = False 
    subset_frac = 0.4

    #preproc related
    #augmentation
    #proba for conservative, weight for aggressive
    
    #conservative
    conservative_aug = []
    #aggressive, OneOf 
    aggressive_aug_proba = 0.5
    aggressive_aug = ['vflip']     
    
    
    vflip = True
    vflip_proba = 0.5
    vflip_weight = 1.0 
    add_gaussian_noise = False 
    add_gaussian_noise_proba = 0.5 
    add_gaussian_noise_weight = 1.0    
    timemask = False
    timemask_proba = 0.35
    timemask_weight = 0.8
    shuffle01 = False
    shuffle01_proba = 0.35
    shuffle01_weight = 0.8
    time_shift = False
    time_shift_left = 96
    time_shift_right = 96
    time_shift_proba = 0.35
    time_shift_weight = 0.4
    
    shift_channel = False
    shift_channel_left = 16
    shift_channel_right = 16
    shift_channel_proba = 0.5
    shift_channel_weight = 1.0
    shift_two_channels = False #tba
    shift_two_channels_proba = 0.5
    shift_two_channels_weight= 1.0
    reduce_SNR = False
    reduce_SNR_ratio = 0.9998
    reduce_SNR_proba = 0.5
    reduce_SNR_weight = 1.0

    time_stretch = False
    divide_std = False 
    shuffle_channels = False    
    pitch_shift = False
    use_mixup = False
    mixup_alpha = 0.1
    cropping = False
    
    #logistic
    seed = 48
    target_size = 1
    target_col = 'target'
    n_fold = 5

    # kaggle_json_path = 'kaggle/kaggle.json'
    
    model_output_folder = output_dir + model_version + "/"
    
    #logger
    print_num_steps=350
    
    #training related
    train_folds = [0]#[0,1,2,3,4]

    weight_decay=0 #1e-4  # Optimizer, default value 0.01
    gradient_accumulation_steps=1 # Optimizer
    scheduler='cosineWithWarmUp' # warm up ratio 0.1 of total steps 
     
    #speedup
    num_workers=0
    non_blocking=False
    amp=True
    use_cudnn = True 
    use_tpu = False
    use_ram = False
    continuous_exp = False





    #augmentation
    conservative_aug = ['vflip','shuffle01',]
    aggressive_aug = []     
    
    #conservative
    vflip = True
    vflip_proba = 0.5
    shuffle01 = True
    shuffle01_proba = 0.5 
 



    
    #CNN structure
    channels = 32
    reduction = 1.0
    stochastic_final_layer_proba = 0.50
    CBAM_SG_kernel_size = 15
