 
#tbs conserv_transform_list_strings etc variables not in scope
#tbs need pythonic way

from itertools import chain, combinations
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import audiomentations as A
from config import Config
import utils
from model import Model
import dataset

class TTA(Dataset):
    def __init__(self, paths, targets, vflip=False, shuffle_channels=False, time_shift=False, 
                 add_gaussian_noise = False,  time_stretch=False,shuffle01=False,timemask=False,
                 shift_channel=False,reduce_SNR=False, ):
        self.paths = paths
        self.targets = targets
        self.vflip = vflip
        self.shuffle_channels = shuffle_channels
        self.time_shift = time_shift
        self.add_gaussian_noise = add_gaussian_noise
        self.time_stretch = time_stretch
        self.shuffle01 = shuffle01
        self.timemask = timemask
        self.shift_channel = shift_channel
        self.reduce_SNR = reduce_SNR
        if time_shift:
            self.time_shift = A.Shift(min_fraction=-Config.time_shift_left*1.0/4096, 
                                      max_fraction=Config.time_shift_right*1.0/4096, p=1,rollover=False)
        if add_gaussian_noise:
            self.add_gaussian_noise = A.AddGaussianNoise(min_amplitude=0.001*0.015, max_amplitude= 0.015*0.015, p=1)

        if timemask:
            self.timemask = A.TimeMask(min_band_part=0.0, max_band_part=0.03, fade=False, p=1.0)
            
        if time_stretch:
            self.time_stretch = A.TimeStretch(min_rate=0.9, max_rate=1.111,leave_length_unchanged=True, p=1)
              
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index] 
        waves = np.load(path)


        if self.vflip:
            waves = -waves
        if self.shuffle_channels:
            np.random.shuffle(waves)
        if self.time_shift:
            waves = self.time_shift(waves, sample_rate=2048)
        if self.add_gaussian_noise:
            waves = self.add_gaussian_noise(waves, sample_rate=2048)
        if self.time_stretch:
            waves = self.time_stretch(waves, sample_rate=2048)
        if self.shuffle01:
            waves[[0,1]] = waves[[1,0]]
        if self.timemask:
            waves = self.timemask(waves, sample_rate=2048)
        # if self.shift_channel:
        #     waves = shift_channel_func(waves, sample_rate=2048)
        # if self.reduce_SNR:
        #     waves = reduce_SNR_func(waves, sample_rate=2048)
        #snr, shift_channel tba
        
        waves = torch.from_numpy(waves) 
        target = torch.tensor(self.targets[index],dtype=torch.float)#device=device,             
        return (waves, target)
    
    
def get_pred(loader,model):
    preds = []
    device = utils.get_device()
    for step, batch in enumerate(loader, 1):
        if step % Config.print_num_steps == 0:
            print("step {}/{}".format(step, len(loader)))
        with torch.no_grad():
            X = batch[0].to(device,non_blocking=Config.non_blocking)
            outputs = model(X).squeeze()
            preds.append(outputs.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

def get_tta_pred(df,model,**transforms):
    data_retriever = TTA(df['file_path'].values, df['target'].values, **transforms)
    loader = DataLoader(data_retriever, 
                            batch_size=Config.batch_size * 2, 
                            shuffle=False, 
                            num_workers=Config.num_workers, pin_memory=True, drop_last=False)
    return get_pred(loader,model)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def save_oof_preds():
    for fold in Config.train_folds:
        print(fold)
        checkpoint = torch.load(f'{Config.model_output_folder}/Fold_{fold}_best_model.pth')
        try:
            oof = pd.read_csv(f'{Config.model_output_folder}/Fold_{fold}_oof_pred.csv')
            oof['pred'] = checkpoint['valid_preds']
            oof.to_csv(f'{Config.model_output_folder}/Fold_{fold}_oof_pred.csv') 
            print('successfully saved oof predictions for Fold: ', fold)   
        except:
            raise RuntimeError('failure in saving predictions for Fold: ', fold)
       



def gen_oof_tta(train_df,conserv_transform_list_strings,aggressive_transform_list_strings):
    model = Model()
    device = utils.get_device()
    for fold in Config.train_folds:
        print('Fold ',fold)
        oof = train_df.query(f"fold=={fold}").copy()
        oof['preds'] = torch.load(f'{Config.model_output_folder}/Fold_{fold}_best_model.pth')['valid_preds']
        oof['file_path'] = train_df['id'].apply(lambda x :dataset.id_2_path(x))
        # display(oof)    
    
        checkpoint = torch.load(f'{Config.model_output_folder}/Fold_{fold}_best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device=device,non_blocking=Config.non_blocking)
        model.eval()
        
        conserv_transform_powerset = list(powerset(conserv_transform_list_strings))
        for transformations in conserv_transform_powerset:
            if transformations:#to avoid double count original
                print("tta_"+('_').join(transformations))
                oof["tta_"+('_').join(transformations)] = get_tta_pred(oof,model,**{transformation:True for transformation in transformations})
            for aggr_transformation in aggressive_transform_list_strings:
                print("tta_"+('_').join(transformations)+'_'+aggr_transformation)
                oof["tta_"+('_').join(transformations)+'_'+aggr_transformation] = get_tta_pred(oof,model,**{transformation:True for transformation in transformations}, **{aggr_transformation:True})
                       
        oof.to_csv(Config.model_output_folder + f"/oof_Fold_{fold}.csv", index=False)

def gen_oof_all_folds():
    oof_all = pd.DataFrame()
    for fold in Config.train_folds:
        oof = pd.read_csv(Config.model_output_folder + f"/oof_Fold_{fold}.csv")
        oof_all = pd.concat([oof_all,oof])
        oof_all.to_csv(Config.model_output_folder + "/oof_all.csv", index=False)
        
def gen_oof_weight(conserv_transform_list_strings,aggressive_transform_list_strings):
    oof_weight  = defaultdict(lambda :1)
    aggr_total_weight = 0
    for trans in aggressive_transform_list_strings:
        aggr_total_weight += getattr(Config(),trans+'_weight')
    
    oof_all_columns = []
    conserv_transform_powerset = list(powerset(conserv_transform_list_strings))
    for transformations in conserv_transform_powerset:
        if transformations:#to avoid double count original
            oof_all_columns.append("tta_"+('_').join(transformations))
        for aggr_transformation in aggressive_transform_list_strings:
            oof_all_columns.append("tta_"+('_').join(transformations)+'_'+aggr_transformation)

    for col in oof_all_columns:
        
        if 'tta_' in col or 'preds' in col: 
            for trans in conserv_transform_list_strings:
                
                if trans in col:
                    oof_weight[col] *= getattr(Config(),trans+'_proba')
                else:
                    oof_weight[col] *= 1-getattr(Config(),trans+'_proba')
                
            flag = False
            for trans in aggressive_transform_list_strings:
                
                if trans in col:
                    oof_weight[col] *= getattr(Config(),trans+'_weight')/aggr_total_weight*Config.aggressive_aug_proba
                    
                    flag = True
            if not flag:
                oof_weight[col] *= (1-Config.aggressive_aug_proba)
                
    return oof_weight


def gen_oof_tta_weighted(oof_weight):#weight consistent with augmentation
    oof_all = pd.read_csv(Config.model_output_folder + "/oof_all.csv")
    oof_all['avg']=0
    
    total_weight = 0 
    for col in oof_all.columns:
        if ('tta_' in col or 'preds' in col): # and 'time_shift' not in col and 'timemask' not in col
            print(col)
            total_weight+=oof_weight[col]
            oof_all['avg'] += oof_all[col]*oof_weight[col]
    oof_all['avg'] /= total_weight
    oof_all.to_csv(Config.model_output_folder + "/oof_all.csv", index=False)
    oof_all[['id','fold','avg']].rename(columns={'id':'id','fold':'fold','avg':'prediction'}).to_csv(Config.model_output_folder + "/oof_final.csv", index=False)


def gen_test_tta(test_df,conserv_transform_list_strings,aggressive_transform_list_strings):
    test_df['target'] = 0  
    model = Model()
    
    for fold in Config.train_folds:
        test_df2 = test_df.copy()
        device = utils.get_device()
        checkpoint = torch.load(f'{Config.model_output_folder}/Fold_{fold}_best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device=device,non_blocking=Config.non_blocking)
        model.eval()
    
        test_df2['preds'+f'_Fold_{fold}'] = get_tta_pred(test_df2,model)
        conserv_transform_powerset = list(powerset(conserv_transform_list_strings))
        for transformations in conserv_transform_powerset:
            if transformations:#to avoid double count original pred
                print("tta_"+('_').join(transformations)+f'_Fold_{fold}')
                test_df2["tta_"+('_').join(transformations)+f'_Fold_{fold}'] = get_tta_pred(test_df2,model,**{transformation:True for transformation in transformations})
            for transformation in aggressive_transform_list_strings:
                print("tta_"+('_').join(transformations)+'_'+transformation+f'_Fold_{fold}')
                test_df2["tta_"+('_').join(transformations)+'_'+transformation+f'_Fold_{fold}'] = get_tta_pred(test_df2,model,**{transformation:True for transformation in transformations}, **{transformation:True})
                   
        test_df2.to_csv(Config.model_output_folder + f"/test_Fold_{fold}.csv", index=False)
        
def gen_test_tta_weighted(test_df, oof_weight):        
    #test TTA weighting    
    test_avg = test_df[['id', 'target']].copy()
    test_avg['target'] = 0
    # print(test_avg.describe())
    
    total_weight = 0
    for fold in Config.train_folds:
        test_weight = oof_weight #same as oof weight
        test_df2 = pd.read_csv(Config.model_output_folder + f"/test_Fold_{fold}.csv")

        for col in test_df2.columns:
            col_weight = col.split('_Fold_')[0]
            if ('tta_' in col or 'preds' in col): 
                total_weight+=test_weight[col_weight]
                test_avg['target'] += test_df2[col]*test_weight[col_weight]
    test_avg['target'] /= total_weight
    print(test_avg.describe())
    print(test_avg["target"].hist(bins=100))
    print(test_avg)
    # print(total_weight)
    test_avg.to_csv(Config.model_output_folder + "/test_avg.csv", index=False)


    # Create Submission File
    test_avg[['id', 'target']].to_csv("./submission.csv", index=False)
    test_avg[['id', 'target']].to_csv(Config.model_output_folder + "/submission.csv", index=False)



