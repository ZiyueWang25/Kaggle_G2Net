class Config:

    #frequently changed 

    model_version = "133rd_V2SD_PL_4ep_2em3lr_32ch_vf_sc01_drop05" 
    model_module = 'V2StochasticDepth'
    use_pseudo_label = True

    pseudo_label_folder = output_dir+"120th_V2_PL_6ep_1em3lr_32ch_vf_s01/"
    
    
    #conservative aug
    conservative_aug = []
    #aggressive, OneOf 
    aggressive_aug_proba = 2.0/3.0
    aggressive_aug = ['vflip','shuffle01']     
    vflip = True
    vflip_weight = 1.0 
    shuffle01 = True
    shuffle01_weight = 0.8
 
    epochs=4
    batch_size = 256
    lr=  2e-3 
    
    #CNN structure
    channels = 32
    stochastic_final_layer_proba = 0.50


