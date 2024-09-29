#!/usr/bin/env python

import tempfile
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

class ChemPropWrapper:
    def __init__(self, y_name):
        self.y_name = y_name

    def validate(self, train, test):
        with tempfile.TemporaryDirectory() as temp_dirname:
            # write the input files
            train.to_csv(f"{temp_dirname}/train.csv")
            test.to_csv(f"{temp_dirname}/test.csv")
            # train the model
            train_args = f"train -s SMILES --task-type regression --data-path {temp_dirname}/train.csv --target-columns {self.y_name} -o {temp_dirname} ".split()
            retcode = subprocess.call(['chemprop', *train_args], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            assert retcode == 0, "ChemProp train failed"      
            # make model predictions
            outfile_name = f"{temp_dirname}/pred.csv"
            predict_args = f"predict -i {temp_dirname}/test.csv -o {outfile_name} --model-path {temp_dirname}/model_0/best.pt -s SMILES ".split()
            retcode = subprocess.call(['chemprop', *predict_args], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            assert retcode == 0, "ChemProp predict failed"
            # read the results
            result_df = pd.read_csv(outfile_name)
            return result_df.pred_0.values

def main():
    df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/datafiles/refs/heads/main/biogen_logS.csv")
    train, test = train_test_split(df)
    chemprop_wrapper = ChemPropWrapper("logS")
    pred = chemprop_wrapper.validate(train, test)
    print(pred)

if __name__ == "__main__":
    main()
        
