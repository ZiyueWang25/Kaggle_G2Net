from src.config import *
from src.dataset import read_data
from src.infer_helper import *
from src.config import read_config, prepare_args


if __name__ == "__main__":
    arg = prepare_args()
    Config = read_config(arg.model_config)
    train_df, test_df = read_data(Config)
    device = get_device()
    seed_torch(seed=Config.seed)
    Config.device=device
    CV_SCORE, oof_all = get_oof_final(train_df, Config)
    test_avg = get_test_avg(CV_SCORE, test_df, Config)
