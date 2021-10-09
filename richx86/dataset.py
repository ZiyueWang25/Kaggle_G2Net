import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from config import Config
from utils import sigmoid
def id_2_path(file_id: str, train=True) -> str:
    if train:
        return "./data/whiten-train-w0/{}.npy".format(file_id)#set path in config
    else:
        return "./data/whiten-test-w0/{}.npy".format(file_id)
    
def read_data():
    train_df = pd.read_csv('./data/training_labels.csv')
    test_df = pd.read_csv('./data/sample_submission.csv')
    if Config.debug:
        Config.epochs = 1
        train_df = train_df.sample(n=50000, random_state=Config.seed).reset_index(drop=True)
    if Config.use_subset:
        train_df = train_df.sample(frac=Config.subset_frac, random_state=Config.seed).reset_index(drop=True)
    train_df['file_path'] = train_df['id'].apply(lambda x :id_2_path(x))
    test_df['file_path'] = test_df['id'].apply(lambda x :id_2_path(x,False))
    #
    
    
    skf = StratifiedKFold(n_splits=Config.n_fold, shuffle=True, random_state=Config.seed)
    splits = skf.split(train_df, train_df["target"])
    train_df['fold'] = -1
    for fold, (train_index, valid_index) in enumerate(splits):
        train_df.loc[valid_index,"fold"] = fold
    return train_df, test_df

def generate_PL(fold,train_df):
    pseudo_label_df = pd.read_csv(Config.pseudo_label_folder + f"test_Fold_{fold}.csv") 
    pseudo_label_df['file_path'] = pseudo_label_df['id'].apply(lambda x :id_2_path(x,False))
    pseudo_label_df["target"] = (pseudo_label_df['target']).apply(sigmoid)

    test_df_2 = pseudo_label_df.copy()
    test_df_2['fold'] = Config.n_fold
    PL_train_df = pd.concat([train_df, test_df_2]).reset_index(drop=True)
    PL_train_df.reset_index(inplace=True, drop=True)

    return PL_train_df

class DataRetriever(Dataset):
    def __init__(self, paths, targets, conserv_transforms=None,aggressive_transform_list=None,aggressive_transform_list_strings=None):
        self.paths = paths
        self.targets = targets
        self.conserv_transforms = conserv_transforms
        self.aggressive_transform_list_strings = aggressive_transform_list_strings
        self.aggressive_transform_list = aggressive_transform_list


        #reading data for fold 0 for fast iteration
        # if Config.continuous_exp and Config.train_folds == [0]:
        #     if Config.use_pseudo_label:
        #         self.data = fold_0_data_PL
        #     else:
        #         self.data = fold_0_data
        # else:
        if Config.use_ram:
            start_time =time.time()
            array_shape = (len(self.paths),3,4096)
            self.data = np.zeros(array_shape,dtype=np.float32)
            for i,path in enumerate(self.paths):
                waves = np.load(path)
                self.data[i,:] = waves            
            print(time.time()-start_time)

                
            # saving Fold 0 data for later use
#         with open('fold_0_data_PL.npy', 'wb') as f:
#             np.save(f, self.data)



    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        if Config.use_ram:
            waves = self.data[index]
        else:
            path = self.paths[index] 
            waves = np.load(path)

           
        if self.conserv_transforms is not None:
            for i,_ in enumerate(self.conserv_transforms):
                transform = self.conserv_transforms[i]
                waves= transform(waves,sample_rate=2048)
            
        if self.aggressive_transform_list_strings:
            if np.random.random()<Config.aggressive_aug_proba:
                n = len(self.aggressive_transform_list_strings)
                probas = np.array([getattr(Config(), f'{agg}_weight') for agg in self.aggressive_transform_list_strings])
                probas /= probas.sum()
                trans_idx = np.random.choice(n,p=probas)
                trans = self.aggressive_transform_list[trans_idx]
                waves = trans(waves,sample_rate=2048)


        waves = torch.from_numpy(waves) 
        target = torch.tensor(self.targets[index],dtype=torch.float)#device=device, 
            
        return (waves, target)

class DataRetrieverTest(Dataset):
    def __init__(self, paths, targets):
        self.paths = paths
        self.targets = targets

        if Config.use_ram:
            array_shape = (len(self.paths),3,4096)
            self.data = np.zeros(array_shape,dtype=np.float32)
            for i,path in enumerate(self.paths):
                waves = np.load(path)
                self.data[i,:] = waves  

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):

        if Config.use_ram:
            waves = self.data[index]
        else:
            path = self.paths[index] 
            waves = np.load(path)
            

        waves = torch.from_numpy(waves) 
        target = torch.tensor(self.targets[index],dtype=torch.float)#device=device, 
            
        return (waves, target)

class DataRetrieverLRFinder(Dataset):
    def __init__(self, paths, targets):
        self.paths = paths
        self.targets = targets


    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        
        path = self.paths[index] 
        waves = np.load(path)
        
              
        waves = torch.from_numpy(waves) 

        target = torch.tensor(self.targets[index],dtype=torch.float)#device=device, 
            
        return (waves, target)
