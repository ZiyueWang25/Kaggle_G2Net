from tqdm import tqdm
from torch.utils.data import DataLoader
from .util import *
from .dataset import *
from .models import getModel
from torch import nn


def get_before_head(x, model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            shape = x.shape
            c = x.view(shape[0] * shape[1], -1)
            c = torch.cat([-c.flip(-1)[:, 4096 - 2049:-1] + 2 * c[:, 0].unsqueeze(-1), c,
                           -c.flip(-1)[:, 1:2049] + 2 * c[:, -1].unsqueeze(-1)], 1)
            avr_spec = model.avr_spec.repeat(shape[0], 1).view(-1, model.avr_spec.shape[-1])
            x = torch.fft.ifft(torch.fft.fft(c * model.window) / avr_spec).real
            x = x.view(shape[0], shape[1], x.shape[-1])
            x = x[:, :, 2048:-2048]
    x0 = [model.ex[0](x[:, 0].unsqueeze(1)), model.ex[0](x[:, 1].unsqueeze(1)),
          model.ex[1](x[:, 2].unsqueeze(1))]
    x1 = [model.conv1[0](x0[0]), model.conv1[0](x0[1]), model.conv1[1](x0[2]),
          model.conv1[2](torch.cat([x0[0], x0[1], x0[2]], 1))]
    x2 = torch.cat(x1, 1)
    return model.conv2(x2)


def get_pred(loader, model, device, use_MC=False, MC_folds=64):
    if isinstance(model, nn.DataParallel):
        model = model.module
    if use_MC:
        model.head[4].train()
        model.head[8].train()
    preds = []
    for step, batch in enumerate(loader, 1):
        if step % 100 == 0:
            print("step {}/{}".format(step, len(loader)))
        with torch.no_grad():
            X = batch[0].to(device)
            if use_MC:
                x2 = get_before_head(X, model)
                preds_MC = [model.head(x2) for i in range(MC_folds)]
                outputs = torch.stack(preds_MC, 0).mean(0)
            else:
                outputs = model(X)
            outputs = outputs.squeeze().sigmoid().cpu().detach().numpy()
            preds.append(outputs)
    predictions = np.concatenate(preds)
    return predictions


def get_tta_pred(df, model, Config, **transforms):
    data_retriever = TTA(df['file_path'].values, df['target'].values, **transforms)
    loader = DataLoader(data_retriever,
                        batch_size=Config.batch_size,
                        shuffle=False,
                        num_workers=Config.num_workers, pin_memory=True, drop_last=False)
    return get_pred(loader, model, Config.device, Config.use_MC, Config.MC_folds)


class TTA(Dataset):
    def __init__(self, paths, targets, use_vflip=False, shuffle_channels=False, time_shift=False,
                 add_gaussian_noise=False, time_stretch=False, shuffle01=False):
        self.paths = paths
        self.targets = targets
        self.use_vflip = use_vflip
        self.shuffle_channels = shuffle_channels
        self.time_shift = time_shift
        self.gaussian_noise = add_gaussian_noise
        self.time_stretch = time_stretch
        self.shuffle01 = shuffle01
        if time_shift:
            self.time_shift = A.Shift(min_fraction=-512 * 1.0 / 4096, max_fraction=-1.0 / 4096, p=1, rollover=False)
        if add_gaussian_noise:
            self.gaussian_noise = A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1)
        if time_stretch:
            self.time_stretch = A.TimeStretch(min_rate=0.9, max_rate=1.111, leave_length_unchanged=True, p=1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        waves = np.load(path)

        if self.use_vflip:
            waves = -waves
        if self.shuffle_channels:
            np.random.shuffle(waves)
        if self.time_shift:
            waves = self.time_shift(waves, sample_rate=2048)
        if self.gaussian_noise:
            waves = self.gaussian_noise(waves, sample_rate=2048)
        if self.time_stretch:
            waves = self.time_stretch(waves, sample_rate=2048)
        if self.shuffle01:
            waves[[0, 1]] = waves[[1, 0]]

        waves = torch.FloatTensor(waves * 1e20)
        target = torch.tensor(self.targets[index], dtype=torch.float)  # device=device,
        return (waves, target)


def get_tta_df(df, model, Config):
    if Config.vflip:
        df["tta_vflip"] = get_tta_pred(df, model, Config, use_vflip=True)
    if Config.shuffle01:
        df["tta_shuffle01"] = get_tta_pred(df, model, Config, shuffle01=True)
    if Config.vflip and Config.shuffle01:
        df["tta_vflip_shuffle01"] = get_tta_pred(df, model, Config, use_vflip=True, shuffle01=True)

    # df["tta_shift"] = get_tta_pred(df,model,time_shift=True)
    # df["tta_vflip_shift"] = get_tta_pred(df,model,use_vflip=True,time_shift=True)
    # df["tta_shift_shuffle01"] = get_tta_pred(df,model,time_shift=True,shuffle01=True)
    # df["tta_vflip_shift_shuffle01"] = get_tta_pred(df,model,use_vflip=True,time_shift=True,shuffle01=True)
    return df


def get_oof_final(train_df, Config):
    oof_all = pd.DataFrame()
    for fold in tqdm(Config.train_folds):
        if Config.model_module == "M3D":
            Config.fold = fold
        model = getModel(Config)
        oof = train_df.query(f"fold=={fold}").copy()
        oof['preds'] = torch.load(f'{Config.model_output_folder}/Fold_{fold}_best_model.pth')['valid_preds']
        if Config.use_swa:
            swa_model = AveragedModel(model)
            checkpoint = torch.load(f'{Config.model_output_folder}/Fold_{fold}_swa_model.pth')
            model = swa_model
            model.load_state_dict(removeDPModule(checkpoint['model_swa_state_dict']))
        else:
            checkpoint = torch.load(f'{Config.model_output_folder}/Fold_{fold}_best_model.pth')
            model.load_state_dict(removeDPModule(checkpoint['model_state_dict']))

        model.to(device=Config.device)
        if Config.use_dp and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()
        oof = get_tta_df(oof, model, Config)
        oof.to_csv(Config.model_output_folder + f"/oof_Fold_{fold}.csv", index=False)
        oof_all = pd.concat([oof_all, oof])
    print("Original:", fast_auc(oof_all['target'], oof_all['preds']))

    for col in oof.columns:
        if "tta" in col:
            print(col, fast_auc(oof_all['target'], oof_all[col]))

    avg_cols = [col for col in oof_all.columns if "tta" in col or "preds" in col]
    oof_all['avg'] = oof_all[avg_cols].mean(axis=1)
    fold_scores = oof_all.groupby("fold").apply(lambda df: fast_auc(df['target'], df['avg']))
    print(fold_scores)
    CV_SCORE = fold_scores.mean()
    print("CV_SCORE:", CV_SCORE)
    oof_all.to_csv(Config.model_output_folder + "/oof_all.csv", index=False)
    oof_final = oof_all[['id', 'fold', 'avg']].rename(columns={'id': 'id', 'fold': 'fold', 'avg': 'prediction'})
    oof_final.to_csv(Config.model_output_folder + f"/oof_final_CV{CV_SCORE * 1e5:.0f}.csv", index=False)
    return CV_SCORE, oof_all


def removeDPModule(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

def get_test_avg(CV_SCORE, test_df, Config):
    test_df['target'] = 0
    test_avg = test_df[['id', 'target']].copy()
    count = 0
    for fold in tqdm(Config.train_folds):
        if Config.model_module == "M3D":
            Config.fold = fold
        model = getModel(Config)
        test_df2 = test_df.copy()
        if Config.use_swa:
            swa_model = AveragedModel(model)
            checkpoint = torch.load(f'{Config.model_output_folder}/Fold_{fold}_swa_model.pth')
            model = swa_model
            model.load_state_dict(removeDPModule(checkpoint['model_swa_state_dict']))
        else:
            checkpoint = torch.load(f'{Config.model_output_folder}/Fold_{fold}_best_model.pth')
            model.load_state_dict(removeDPModule(checkpoint['model_state_dict']))
        model.to(device=Config.device)
        if Config.use_dp and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()
        test_df2['preds' + f'_Fold_{fold}'] = get_tta_pred(test_df2, model, Config)
        test_df2 = get_tta_df(test_df2, model, Config)
        test_df2.to_csv(Config.model_output_folder + f"/test_Fold_{fold}.csv", index=False)
        for col in test_df2.columns:
            if "tta" in col or 'preds' in col:
                count += 1
                test_avg['target'] += test_df2[col]
    test_avg['target'] /= count
    test_avg[['id', 'target']].to_csv(Config.model_output_folder + f"/submission_CV{CV_SCORE * 1e5:.0f}.csv",
                                      index=False)
    print(Config.model_output_folder + f"/submission_CV{CV_SCORE * 1e5:.0f}.csv")
    return test_avg
