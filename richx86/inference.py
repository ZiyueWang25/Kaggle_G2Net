from oof_and_tta import save_oof_preds,gen_oof_tta,gen_oof_all_folds,gen_oof_weight,gen_oof_tta_weighted,gen_test_tta,gen_test_tta_weighted
from augmentation import augmentations
import dataset

(conserv_transform_list,
         aggressive_transform_list, 
         conserv_transform_list_strings, 
         aggressive_transform_list_strings) = augmentations()

train_df,test_df = dataset.read_data()

save_oof_preds()
gen_oof_tta(train_df,conserv_transform_list_strings,aggressive_transform_list_strings)
gen_oof_all_folds()
oof_weight = gen_oof_weight(conserv_transform_list_strings,aggressive_transform_list_strings)
gen_oof_tta_weighted(oof_weight)
gen_test_tta(test_df,conserv_transform_list_strings,aggressive_transform_list_strings)
gen_test_tta_weighted(test_df, oof_weight)