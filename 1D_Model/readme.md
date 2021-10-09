# Reproduce 1D Model Result

Below you can find a outline of how to reproduce our 1D Model solution for the G2Net-Gravitational-Wave-Detection competition.

## Folder CONTENTS

- dataset/ 
  - Models/: original kaggle model upload - contains model weights, OOF prediction, test prediction
  - PL_fold/: OOF fold prediction for pseudo labeling
  - hack-lb/: contains oof fold prediction for hacking LB in the end
  - ensemble-0916/: contains model oof prediction, test prediction throughout the ensembling phase
  - avr_w0.pth: average ASD using target 0 samples for preprocessing
  - design_curves_tukey_0.2.npy: design curve from Anjum
  - GW_sim_300k.pkl: simulated Gravitational wave.
- notebooks/:
  - `stacking.ipynb`: for stacking, use hacking-lb method
  - `SyntheticSignal.ipynb`: for GW simulation
  - `get_avg_w0.ipynb`: for avg_w0 generation
- src/:
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
  - `util.py`: utility functions
- `infer.py`: inference interface
- `train.py`: training interface
- `reproduce_trian.sh`: script file for model reproduction
- `reproduce_infer.sh`: script file for model prediction (OOF, test) reproduction

## HARDWARE: (The following specs were used to create the original solution)

1 x NVIDIA Tesla A100 or 2 x NVIDIA GeForce RTX 2080Ti.

## DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)

1. If there is no `dataset` folder in the current folder, please download it from the google drive and put it inside this folder

## Below are the shell commands used in each step, as run from the top level directory

1. cd dataset/
2. kaggle competitions download -c g2net-gravitational-wave-detection
3. unzip -q g2net-gravitational-wave-detection
4. cd ../
5. python reproduce.sh

## MODEL BUILD: There are two options to produce the solution.

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
