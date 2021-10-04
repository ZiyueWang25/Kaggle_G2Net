import os
from argparse import ArgumentParser
import yaml
import torch
import pickle5 as pickle

class BaseConfig:
    #logistic
    seed = 48
    target_size = 1
    gdrive = './input/'
    output_dir = "./output_model/"
    pseudo_label_folder = "./PL_fold/"
    use_checkpoint = False
    prev_model_folder = None    
    debug = False
    use_subset = False 
    subset_frac = 0.4
    # augmentation    
    vflip = False
    shuffle01 = False
    time_shift = False
    time_stretch = False
    shuffle_channels = False  # need normalization first
    add_gaussian_noise = False # need normalization first
    timemask = False
    shift_channel = False    
    pitch_shift = False
    use_mixup = False
    mixup_alpha = 0.1
    cropping = False
    
    use_MC=False
    MC_folds=64    
    # logger
    print_num_steps=350
    use_wandb=True
    # training related
    train_folds = [0,1,2,3,4]
    epochs = 6
    batch_size = 256    
    lr= 1e-2
    weight_decay= 1e-4
    gradient_accumulation_steps=1
    scheduler='cosineWithWarmUp' 
    # SWA
    use_swa = False
    swa_lr_ratio = 0 # in terms of max lr
    swa_lr = 0
    swa_start_step_epoch = 3
    swa_anneal_ratio =  999, # 999 means anneal til the end of the training
    # speedup
    num_workers=8
    use_cudnn = True 
    use_dp=True  # dataparallel

    
class V2_Config(BaseConfig):
    model_name = 'TCNN'
    model_version="main_82nd_V2_c16_sGW_vflip_sc01_PL_script"
    model_module="V2"
    vflip=True
    shuffle01=True
    use_MC=True
    
    lr=7e-3
    checkpoint_folder=True
    epochs=6
    wandb_name = 'main_112th_V2SD_sGW_vflip_sc01_PL_script'
    use_dp=False     

class V2_Config_pretrain(V2_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=2
    wandb_name = V2_Config.model_version + "_pretrain"     

class V2SD_Config(BaseConfig):
    model_name = 'TCNN'
    model_version="main_112th_V2SD_sGW_vflip_sc01_PL_script"
    model_module="V2SD"
    vflip=True
    shuffle01=True
    
    checkpoint_folder=True
    epochs=6
    wandb_name = 'main_112th_V2SD_sGW_vflip_sc01_PL_script'
    use_dp=False    

class V2SD_Config_pretrain(V2SD_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=2
    wandb_name = V2SD_Config.model_version + "_pretrain"     
    

class resnet34_Config(BaseConfig):
    model_name = 'Model_2D'
    model_version = "resnet34-sGW2ep-PL-sc01-5ep_script" 
    model_2D_encoder = 'resnet34'
    
    shuffle01=True
    checkpoint_folder=True
    epochs=4
    wandb_name = 'resnet34-sGW2ep-PL-sc01-5ep_script'
    use_dp=True    

class resnet34_Config_pretrain(resnet34_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=2
    wandb_name = resnet34_Config.model_version + "_pretrain"     


class M3D_Config(BaseConfig):
    model_name = '1D2D'
    model_version="1D2D-Pretrained-1ep1e2-1ep1e5_script"
    model_module="M3D"
    
    vflip=True
    shuffle01=True
    use_MC=False
    MC_folds=64
    
    lr=5e-3
    checkpoint_folder=False
    epochs=1
    wandb_name = '1D2D-Pretrained-1ep1e2-1ep1e5_script'
    use_dp=True
    
    
def read_config(name):
    print("Read Configuration")
    Config = None
    if name == "V2SD":
        Config = V2SD_Config
    elif name == "V2SD_pretrain":
        Config = V2SD_Config_pretrain
    elif name == "V2":
        Config = V2_Config
    elif name == "V2_pretrain":
        Config = V2_Config_pretrain
    elif name == "resnet34":
        Config = resnet34_Config
    elif name == "resnet34_pretrain":
        Config = resnet34_Config_pretrain
    elif name == "M3D":
        Config = M3D_Config
        
    if Config is None:
        print(f"Configuration {name} is not found")
        return Config

    Config.model_output_folder = Config.output_dir + Config.model_version + "/"
    if Config.checkpoint_folder:
        Config.checkpoint_folder = Config.output_dir + Config.prev_model_folder + "/" \
                                   if Config.prev_model_folder is not None else Config.model_output_folder
    
    if Config.model_output_folder and not os.path.exists(Config.model_output_folder): 
        os.makedirs(Config.model_output_folder)
    torch.backends.cudnn.benchmark = Config.use_cudnn 
    print("Model Output Folder:", Config.model_output_folder)
    return Config    
    
def read_model_dict(model_module):
    print(model_module)
    if model_module == "V2":
        model_dict = dict(
            model_module=model_module,
            channels=32,
            use_raw_wave=True,
        )
    if model_module == "V2SD":
        model_dict = dict(
            model_module=model_module,
            channels=32,
            proba_final_layer=0.8,
            use_raw_wave=True,
        )
    if model_module == "resnet34":
        model_dict = dict(
            model_module=model_module,
            encoder="resnet34",
            use_raw_wave=True,
        )
    if model_module == "M3D":
        model_dict = dict(
            model_module=model_module,
            model_1D = 'V2',
            model_1D_emb=128,
            model_1D_pretrain_dir = "./output_model/main_82nd_V2_c16_sGW_vflip_sc01_PL_script/",

            model_2D = 'resnet34',
            model_2D_emb=128,
            model_2D_pretrain_dir = "./output_model/resnet34-sGW2ep-PL-sc01-5ep_script/",
        )
    return model_dict

def prepare_args():
    parser = ArgumentParser()
    parser.add_argument('--model_module', type=str, help='model module')
    parser.add_argument('--model_config', type=str, help='configuration name for this run')
    args = parser.parse_args()
    return args
    