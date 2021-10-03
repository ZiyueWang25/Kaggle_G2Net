# g2net-gravitational-wave-detection

This folder contains [datasaurus's](https://www.kaggle.com/anjum48) part of the G2Net Gravitational Wave Detection solution.
# Setup
Edit `src/config.py` to reflect the `INPUT_PATH` and `OUTPUT_PATH` locations on your machine:
* `INPUT_PATH` is where the competition train & test data resides as well as any pseudolabelling files
* `OUTPUT_PATH` is where the outputted model weights will be saved. If you have the model weights, put them in this folder.

# Training
To train a single model using a config listed in `hyperparams.yml` run:
```
python train.py --config <config_name>
```
To run a 5-fold cross validation, using 5 different seeds, use the shell script `train.sh`. This script will also run `infer.py` and 
generate out-of-fold (OOF) predictions for stacking models.
```
sh train.sh <config_name>
```
Note that the configurations are set for training on a machine with 2 GPUs. You may need to edit `hyperparams.yml` to reflect your hardware setup.

# Reproducing Kaggle models
To reproduce the model weights used in the final submission run:
```
sh reproduce_train.sh 
```
Note that a CSV file with the pseudolabels will need to be in the `INPUT_PATH` for the pesudolabelled models (e.g. `submission_power2_weight.csv`).

To reproduce the submission files (and OOF predictions), copy all the model weights to the `OUTPUT_PATH` and run:
```
sh reproduce_infer.sh 
```