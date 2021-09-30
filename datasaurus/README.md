# g2net-gravitational-wave-detection
https://www.kaggle.com/c/g2net-gravitational-wave-detection

# Setup
Edit `src/config.py` to reflect the input and output locations on your machine

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

