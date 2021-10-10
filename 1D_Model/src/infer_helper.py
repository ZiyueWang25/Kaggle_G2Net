from itertools import chain, combinations
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from .util import *
from .dataset import *
from .TTA import *
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
        if step % 500 == 0:
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
                        batch_size=Config.batch_size * 2,
                        shuffle=False,
                        num_workers=Config.num_workers, pin_memory=True, drop_last=False)
    return get_pred(loader, model, Config.device, Config.use_MC, Config.MC_folds)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_tta_df(df, model, Config):
    if Config.cons_func_names or Config.aggr_func_names:
        conserv_transform_powerset = list(powerset(Config.cons_func_names))
        for transformations in conserv_transform_powerset:
            if transformations:  # to avoid double count original
                print("tta_" + ('_').join(transformations))
                df["tta_" + ('_').join(transformations)] = get_tta_pred(df, model,
                                                                         **{transformation: True for transformation
                                                                            in transformations})
            for aggr_transformation in Config.aggr_func_names:
                print("tta_" + ('_').join(transformations) + '_' + aggr_transformation)
                df["tta_" + ('_').join(transformations) + '_' + aggr_transformation] = get_tta_pred(df, model, **{
                    transformation: True for transformation in transformations}, **{aggr_transformation: True})
    else:
        if Config.vflip:
            df["tta_vflip"] = get_tta_pred(df, model, Config, vflip=True)
        if Config.shuffle01:
            df["tta_shuffle01"] = get_tta_pred(df, model, Config, shuffle01=True)
        if Config.vflip and Config.shuffle01:
            df["tta_vflip_shuffle01"] = get_tta_pred(df, model, Config, vflip=True, shuffle01=True)
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
    oof_all.to_csv(Config.model_output_folder + "/oof_all.csv", index=False)

    for col in oof.columns:
        if "tta" in col:
            print(col, fast_auc(oof_all['target'], oof_all[col]))
    if Config.cons_func_names or Config.aggr_func_names:
        oof_weight = gen_oof_weight(Config)
        oof_all = gen_oof_tta_weighted(oof_all, oof_weight)
    else:
        avg_cols = [col for col in oof_all.columns if "tta" in col or "preds" in col]
        oof_all['avg'] = oof_all[avg_cols].mean(axis=1)
    fold_scores = oof_all.groupby("fold").apply(lambda df: fast_auc(df['target'], df['avg']))
    print(fold_scores)
    cv_score = fold_scores.mean()
    print("CV_SCORE:", cv_score)
    oof_final = oof_all[['id', 'fold', 'avg']].rename(columns={'id': 'id', 'fold': 'fold', 'avg': 'prediction'})
    oof_final.to_csv(Config.model_output_folder + f"/oof_final_CV{cv_score * 1e5:.0f}.csv", index=False)
    return cv_score, oof_all


def gen_oof_weight(Config):
    if Config.aggr_func_names is None and Config.cons_func_names is None:
        return None
    oof_weight = defaultdict(lambda: 1)
    aggr_total_weight = 0
    for trans in Config.aggr_func_names:
        aggr_total_weight += getattr(Config(), trans + '_weight')
    # get columns
    oof_all_columns = []
    conserv_transform_powerset = list(powerset(Config.cons_func_names))
    for transformations in conserv_transform_powerset:
        if transformations:  # to avoid double count original
            oof_all_columns.append("tta_" + '_'.join(transformations))
        for aggr_transformation in Config.aggr_func_names:
            oof_all_columns.append("tta_" + '_'.join(transformations) + '_' + aggr_transformation)

    for col in oof_all_columns:
        if 'tta_' in col or 'preds' in col:
            for trans in Config.cons_func_names:
                oof_weight[col] *= getattr(Config(), trans + '_proba') if (trans in col) else\
                                   (1 - getattr(Config(), trans + '_proba'))
            flag = False
            for trans in Config.aggr_func_names:
                if trans in col:
                    oof_weight[col] *= getattr(Config(), trans + '_weight') \
                                       / aggr_total_weight * Config.aggressive_aug_proba
                    flag = True
            if not flag:
                oof_weight[col] *= (1 - Config.aggressive_aug_proba)
    return oof_weight


def gen_oof_tta_weighted(oof_all, oof_weight):  # weight consistent with augmentation
    oof_all['avg'] = 0
    total_weight = 0
    for col in oof_all.columns:
        if ('tta_' in col or 'preds' in col):  # and 'time_shift' not in col and 'timemask' not in col
            print(col)
            total_weight += oof_weight[col]
            oof_all['avg'] += oof_all[col] * oof_weight[col]
    oof_all['avg'] /= total_weight
    return oof_all


def removeDPModule(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def get_test_avg(CV_SCORE, test_df, Config):
    test_df['target'] = 0
    test_avg = test_df[['id', 'target']].copy()
    test_weight = gen_oof_weight(Config)
    total_weight = 0
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
                col_weight = 1 if test_weight is None else test_weight[col]
                total_weight += col_weight
                test_avg['target'] += test_df2[col] * col_weight
    test_avg['target'] /= total_weight
    test_avg[['id', 'target']].to_csv(Config.model_output_folder + f"/submission_CV{CV_SCORE * 1e5:.0f}.csv",
                                      index=False)
    print(test_avg['target'].describe())
    print(Config.model_output_folder + f"/submission_CV{CV_SCORE * 1e5:.0f}.csv")
    return test_avg
