import pickle5 as pickle
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .util import id_2_path_wave


class DataRetriever(Dataset):
    def __init__(self, paths, targets, transforms=None, synthetic=None):
        self.paths = paths
        self.targets = targets
        self.transforms = transforms
        self.synthetic = synthetic
        self.synthetic_keys = list(self.synthetic.keys()) if synthetic is not None else None
        self.neg_idxes = [i for i, t in enumerate(targets) if t == 0]                

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        target = self.targets[index]
        path = self.paths[index]
        if target > 0 and (self.synthetic is not None): 
            path = self.paths[random.choice(self.neg_idxes)]                
        waves = np.load(path)
        
        if np.random.random()<0.5:
            waves[[0,1]]=waves[[1,0]]
        if np.random.random()<0.5:
            waves = -waves              
        if self.transforms is not None:
            waves= self.transforms(waves,sample_rate=2048)
        x = torch.FloatTensor(waves*1e20)
        if target > 0 and (self.synthetic is not None):
            w = torch.FloatTensor((self.synthetic[random.choice(self.synthetic_keys)]))
            w = w * max(random.gauss(3.6,1),1)
            shift_place = 512+384-random.randrange(0,768)
            w = np.pad(w[:,shift_place:],[(0,0),(0,shift_place)],mode='constant')
            x += w
        target = torch.tensor(target,dtype=torch.float)
        return (x, target)


class DataRetrieverTest(Dataset):
    def __init__(self, paths, targets, transforms=None):
        self.paths = paths
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index] 
        waves = np.load(path)
        target = self.targets[index]
        if self.transforms is not None:
            waves= self.transforms(waves,sample_rate=2048)
        x = torch.FloatTensor(waves*1e20)
        target = torch.tensor(target,dtype=torch.float)
        return (x, target)
    
def generate_PL(fold, train_df,test_df, Config):
    if Config.PL_folder is None:
        return train_df
    pseudo_label_df = pd.read_csv(Config.PL_folder + f"test_Fold_{fold}.csv")     
    pseudo_label_df['file_path'] = pseudo_label_df['id'].apply(lambda x :id_2_path_wave(x, Config.gdrive, False))
    pseudo_label_df["target"] = pseudo_label_df[f'preds_Fold_{fold}']
    test_df_2 = pseudo_label_df.copy()
    test_df_2['fold'] = -1
    PL_train_df = pd.concat([train_df, test_df_2]).reset_index(drop=True)
    PL_train_df.reset_index(inplace=True, drop=True)
    return PL_train_df
    

def read_synthetic(Config):
    print("Read Synthetic Data")
    with open(Config.gdrive+'/GW_sim_300k.pkl', 'rb') as handle:
        SIGNAL_DICT = pickle.load(handle)
    return SIGNAL_DICT
    
    
def read_data(Config):
    print("Read Data")
    train_df = pd.read_csv(Config.gdrive + '/training_labels.csv')
    test_df = pd.read_csv(Config.gdrive + '/sample_submission.csv')

    if Config.debug:
        Config.epochs = 1
        train_df = train_df.sample(n=50000, random_state=Config.seed).reset_index(drop=True)    
    if Config.use_subset:
        train_df = train_df.sample(frac=Config.subset_frac, random_state=Config.seed).reset_index(drop=True)

    train_df['file_path'] = train_df['id'].apply(lambda x :id_2_path_wave(x, Config.gdrive, True))
    test_df['file_path'] = test_df['id'].apply(lambda x :id_2_path_wave(x,Config.gdrive, False))

    print("StratifiedKFold")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.seed)
    splits = skf.split(train_df, train_df["target"])
    train_df['fold'] = -1
    for fold, (train_index, valid_index) in enumerate(splits):
        train_df.loc[valid_index,"fold"] = fold
    train_df.groupby('fold')['target'].apply(lambda s: s.value_counts(normalize=True))
    return train_df, test_df
    