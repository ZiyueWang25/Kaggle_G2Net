from src.util import *
from src.dataset import read_data, read_synthetic
from src.train_helper import training_loop 
from src.config import read_config, prepare_args

if __name__ == "__main__":
    arg = prepare_args()
    Config = read_config(arg.model_config)
    train_df, test_df = read_data(Config)
    device = get_device()
    seed_torch(seed=Config.seed)
    
    Config = read_config(arg.model_config+"_pretrain")
    if Config is not None:
        Config.device = device
        print("pretraining with synthetic data")
        SIGNAL_DICT = read_synthetic(Config)
        training_loop(train_df, test_df, Config, SIGNAL_DICT)

    Config = read_config(arg.model_config)
    Config.device = device
    training_loop(train_df, test_df, Config, None)
