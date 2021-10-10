# Reproduce 1D Model Result

Below you can find a outline of how to reproduce our 1D Model solution for the G2Net-Gravitational-Wave-Detection competition.

## Folder CONTENTS

- dataset/ 
  - Models/: original kaggle model upload - contains model weights, OOF prediction, test prediction
  - PL_fold/: OOF fold prediction for pseudo labeling
  - 120th_V2_PL_6ep_1em3lr_32ch_vf_s01_5Fold/: PL from 120th model
  - main_35th_GeM_vflip_shuffle01_5fold/: PL from 35th model
  - main_112th_V2SD_PL_6ep_5Fold/: PL from 112nd model
  - hack-lb/: contains oof fold prediction for hacking LB in the end
  - ensemble-0916/: contains model oof prediction, test prediction throughout the ensembling phase
  - avr_w0.pth: average ASD using target 0 samples for preprocessing
  - design_curves_tukey_0.2.npy: design curve from Anjum
  - GW_sim_300k.pkl: simulated Gravitational wave.
- notebooks/:
  - `stacking.ipynb`: for stacking, use hacking-lb method
  - `SyntheticSignal.ipynb`: for GW simulation
  - `get_avg_w0.ipynb`: for avg_w0 generation
  - `generate_whiten_wave.ipynb`: for whiten wave generation: `../dataset/whiten-train-w0/` and `../dataset/whiten-test-w0/`
  - Richard_Models/: folder contains the original notebook for model generation from Richard
- src/:
  - `augmentation.py`: augmentation functions 
  - `config.py`: Model configuration
  - `dataset.py`: dataset preparation
  - `infer_helper.py`: helper functions for inference
  - `loss.py`: related loss functions
  - `lrfinder.py`: learning rate finder class
  - `models.py`: interface to decide which model to use
  - `models_1d.py`: 1D model structure
  - `models_2d.py`: 2D model structure
  - `models_3d.py`: 3D model structure
  - `optim.py`: optimizer class
  - `train_helper.py`: helper functions for training
  - `TTA.py`: class for test time augmentation
  - `util.py`: utility functions
- `infer.py`: inference interface
- `train.py`: training interface
- `reproduce_trian.sh`: script file for model reproduction
- `reproduce_infer.sh`: script file for model prediction (OOF, test) reproduction

## HARDWARE: (The following specs were used to create the original solution)

1 x NVIDIA Tesla A100 or 2 x NVIDIA GeForce RTX 2080Ti.

## DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)

1. If there is no `data` folder in the parent folder, please 
   1. download it from the google drive 
   2. put it inside parent folder
   3. rename the folder as `data` folder

## Below are the shell commands used in each step, as run from current directory

1. Download kaggle competition data
   1. `cd ../data/`
   2. `kaggle competitions download -c g2net-gravitational-wave-detection`
   3. `unzip -q g2net-gravitational-wave-detection`
2. Generate whiten wave from competition data
   1. run notebook `1D_Model/notebooks/generate_whiten_wave.ipynb)`
3. Reproduce Result
   1. `cd ../1D_Model/`
   2. reproduce models:`./reproduce_train.sh`  
   3. reproduce predictions: `./reproduce_infer.sh`

## MODEL BUILD: There are three options to produce the solution.

1) very fast prediction: run `notebook/stacking.ipynb`
    a) runs in 2 hours
    b) uses precomputed predictions
2) ordinary prediction: run `python reproduce_infer.sh`
    a) expect this to run for 6 hours
    b) uses pretrained models
3) retrain models: run `python reproduce_train.sh`
    a) expect this to run about a week
    b) trains all models from scratch
    c) follow this with (2) to produce entire solution from scratch
