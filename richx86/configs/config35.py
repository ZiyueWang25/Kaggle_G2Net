class Config:

    #frequently changed 
    model_version = "main_35th_GeM_vflip_shuffle01_5fold" 
    model_module = 'Model1DCNNGEM'
    use_pseudo_label = False

    #augmentation
    conservative_aug = ['vflip','shuffle01',]
    aggressive_aug = []     
    
    #conservative
    vflip = True
    vflip_proba = 0.5
    shuffle01 = True
    shuffle01_proba = 0.5 

    epochs = 12
    batch_size = 256

    lr= 5e-3
    
    #CNN structure
    channels = 32
    