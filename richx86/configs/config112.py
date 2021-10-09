class Config:

    #frequently changed 
    model_version = "main_112th_V2SD_PL_6ep_5Fold" 
    model_module = 'V2StochasticDepth'
    use_pseudo_label = True


    #augmentation
    conservative_aug = ['vflip','add_gaussian_noise',]
    aggressive_aug = []     
    #conservative
    
    vflip = True
    vflip_proba = 0.5 
    add_gaussian_noise = True 
    add_gaussian_noise_proba = 0.5 
   
    
    
    pseudo_label_folder = output_dir+"main_35th_GeM_vflip_shuffle01_5fold/"
        
    epochs = 6
    batch_size = 256
    lr=  2e-3 
    
    
    #CNN structure
    channels = 32
    stochastic_final_layer_proba = 0.8