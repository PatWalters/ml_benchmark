#!/usr/bin/env python

import polaris as po
from chemprop_wrapper import ChemPropWrapper
from lgbm_wrapper import LGBMMorganCountWrapper, LGBMPropWrapper
import useful_rdkit_utils as uru

def main():
    ds = po.load_dataset("polaris/adme-fang-1")
    y_list = [x for x in ds.columns if x.startswith("LOG_SOL")]
    for y in y_list:
        # get the dataframe from the Polaris dataset
        df = ds[:].dropna(subset=y).copy()
        df.rename(columns={"smiles" : "SMILES"},inplace=True)
        print(f"Processing {y} with {len(df)} records")
        model_list = [("chemprop",ChemPropWrapper),("lgbm_morgan", LGBMMorganCountWrapper),("lgbm_prop",LGBMPropWrapper)]
        group_list = [("random", uru.get_random_clusters),("butina", uru.get_butina_clusters)]
        result_df = uru.cross_validate(df,model_list,y,group_list)
        result_df.to_csv(f"{y}_results.csv",index=False)


if __name__ == "__main__":
    main()
    
