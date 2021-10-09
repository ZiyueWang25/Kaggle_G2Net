import os
import gc
import numpy as np
import pandas as pd

# import IPython.display
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as torch_functional
from torch.optim import AdamW#,Adam, SGD


# import torch_xla
# import torch_xla.core.xla_model as xm

from config import Config
import utils 
from augmentation import augmentations
from trainer import Trainer
import dataset
from dataset import DataRetriever,DataRetrieverTest
from model import Model


# np.set_printoptions(precision=5, suppress=True) 
torch.backends.cudnn.benchmark = Config.use_cudnn 
if not os.path.exists(Config.output_dir):
    os.mkdir(Config.output_dir)
if not os.path.exists(Config.model_output_folder):
    os.mkdir(Config.model_output_folder)






def training_loop(train_df, use_checkpoint=Config.use_checkpoint):
    folds_val_score = []
    original_train_df = train_df.copy()#for PL
    (conserv_transform_list,
         aggressive_transform_list, 
         conserv_transform_list_strings, 
         aggressive_transform_list_strings) = augmentations()
    device=utils.get_device()
    for fold in range(Config.n_fold): 
        if Config.use_pseudo_label:
            PL_train_df = dataset.generate_PL(fold,original_train_df.copy())   
            train_df = PL_train_df
        train_index, valid_index = train_df.query(f"fold!={fold}").index, train_df.query(f"fold=={fold}").index #fold means fold_valid 
        print('Fold: ', fold)
        if fold not in Config.train_folds:
            print("skip")
            continue
        train_X, valid_X = train_df.loc[train_index], train_df.loc[valid_index]
        valid_labels = train_df.loc[valid_index,Config.target_col].values

        oof = pd.DataFrame()
        oof['id'] = train_df.loc[valid_index,'id']
        oof['id'] = valid_X['id'].values.copy()
        oof = oof.reset_index()

        oof['target'] = valid_labels
        
        oof.to_csv(f'{Config.model_output_folder}/Fold_{fold}_oof_pred.csv')
        
        
        
        print('training data samples, val data samples: ', len(train_X) ,len(valid_X))
        train_data_retriever = DataRetriever(train_X["file_path"].values, train_X["target"].values, 
                                             conserv_transforms=conserv_transform_list,
                                             aggressive_transform_list = aggressive_transform_list,
                                             aggressive_transform_list_strings=aggressive_transform_list_strings)
        valid_data_retriever = DataRetrieverTest(valid_X["file_path"].values, valid_X["target"].values)        
        train_loader = DataLoader(train_data_retriever,
                                  batch_size=Config.batch_size, 
                                  shuffle=True, 
                                  num_workers=Config.num_workers, pin_memory=True, drop_last=False)
        valid_loader = DataLoader(valid_data_retriever, 
                                  batch_size=Config.batch_size, 
                                  shuffle=False, 
                                  num_workers=Config.num_workers, pin_memory=True, drop_last=False)

        model = Model()
        model.to(device,non_blocking=Config.non_blocking)
        optimizer = AdamW(model.parameters(), lr=Config.lr,eps=1e-04, weight_decay=Config.weight_decay, amsgrad=False) 
        scheduler = utils.get_scheduler(optimizer, len(train_X))
        best_valid_score = -np.inf
        if use_checkpoint:
            print("Load Checkpoint")
            checkpoint = torch.load(f'{Config.model_output_folder}/Fold_{fold}_best_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_valid_score = float(checkpoint['best_valid_score'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        
        criterion = torch_functional.binary_cross_entropy_with_logits
        

        trainer = Trainer(
            model, 
            device, 
            optimizer, 
            criterion,
            scheduler,
            valid_labels,
            best_valid_score,
            fold
        )

        trainer.fit(
            epochs=Config.epochs, 
            train_loader=train_loader, 
            valid_loader=valid_loader,
            save_path=f'{Config.model_output_folder}/Fold_{fold}_',
        )
        folds_val_score.append(trainer.best_valid_score)
        # del train_data_retriever

    print('folds score:', folds_val_score)
    print("Avg: {:.5f}".format(np.mean(folds_val_score)))
    print("Std: {:.5f}".format(np.std(folds_val_score)))
    
    

utils.seed_everything(seed=Config.seed)  
train_df,test_df = dataset.read_data()

try:
#     %lprun -f DataRetriever.__getitem__ -f Trainer.train_epoch -f Trainer.fit -f Trainer.valid_epoch training_loop() 
    training_loop(train_df,Config.use_checkpoint)
except RuntimeError as e:
    gc.collect()
    torch.cuda.empty_cache()   
    print(e)
    
    
