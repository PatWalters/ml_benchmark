#!/usr/bin/env python

from typing import List, Callable, Tuple

import pandas as pd
import useful_rdkit_utils as uru
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm.auto import tqdm

from lgbm_wrapper import LGBMPropWrapper, LGBMMorganCountWrapper
from chemprop_wrapper import ChemPropWrapper, ChemPropRDKitWrapper
import polaris as po


def cross_validate(df: pd.DataFrame,
                   model_list,
                   y_col: str,
                   group_list: List[Tuple[str, Callable]],
                   metric_list: List[Tuple[str, Callable]],
                   n_outer: int = 5,
                   n_inner: int = 5) -> List[dict]:
    metric_vals = []
    fold_df_list = []
    input_cols = df.columns
    for i in tqdm(range(0, n_outer), leave=False):
        kf = uru.GroupKFoldShuffle(n_splits=n_inner, shuffle=True)
        for group_name, group_func in group_list:
            # assign groups based on cluster, scaffold, etc
            current_group = group_func(df.SMILES)
            for j, [train_idx, test_idx] in enumerate(
                    tqdm(kf.split(df, groups=current_group), total=n_inner, leave=False)):
                fold = i * n_outer + j
                train = df.iloc[train_idx].copy()
                test = df.iloc[test_idx].copy()

                train['dset'] = 'train'
                test['dset'] = 'test'
                train['group'] = group_name
                test['group'] = group_name
                train['fold'] = fold
                test['fold'] = fold

                for model_name, model_class in model_list:
                    model = model_class(y_col)
                    pred = model.validate(train, test)

                    test[model_name] = pred
                    metric_dict = {'group': group_name, 'model': model_name, 'fold': fold}
                    for metric_name, metric_func in metric_list:
                        metric_dict[metric_name] = metric_func(test[y_col], pred)
                    metric_vals.append(metric_dict)
                fold_df_list.append(pd.concat([train, test]))
    output_cols = list(input_cols) + ['dset','group','fold'] + [x[0] for x in model_list]
    pd.concat(fold_df_list)[output_cols].to_csv(f"{y_col}_folds.csv", index=False)
    return metric_vals


def cross_validate_polaris(dataset_name, y_list):
    ds = po.load_dataset(dataset_name)
    for y in y_list:
        df = ds[:].dropna(subset=y).copy()
        df.rename(columns={"smiles" : "SMILES"},inplace=True)
        print(f"Processing {y} with {len(df)} records")
        model_list = [("lgbm_morgan", LGBMMorganCountWrapper),("lgbm_desc", LGBMPropWrapper),
                      ("chemprop",ChemPropWrapper),("chemprop_rdkit",ChemPropRDKitWrapper)]
        group_list = [("butina", uru.get_butina_clusters), ("random", uru.get_random_split),
                  ("scaffold", uru.get_bemis_murcko_clusters)]
        metric_list = [("R2", r2_score), ("MAE", mean_absolute_error)]
        result_list = cross_validate(df, model_list, y, group_list, metric_list, 5, 5)
        result_df = pd.DataFrame(result_list)
        print(result_df.head())


if __name__ == "__main__":
    dataset_name = "polaris/adme-fang-1"
    dataset_columns = ['LOG_HLM_CLint', 'LOG_RLM_CLint', 'LOG_MDR1-MDCK_ER', 'LOG_HPPB', 'LOG_RPPB', 'LOG_SOLUBILITY']
    cross_validate_polaris(dataset_name, dataset_columns)
