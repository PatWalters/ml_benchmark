# ml_benchmark

A set of routines for benchmarking ML methods in drug discovery. The script **run_benchmark.py** can be used to run benchmarks.  All that is necessary for the script is a wrapper class that supports a **validate** method. The wrapper class is instantiated with the name of column to be predicted.  The validate method takes dataframes containing training and test sets as input and returns a list of predicted values for the test set. For examples of wrapper classes see **chemprop_wrapper.py** and **lgbm_wrapper.py**. 

```python
df = pd.read_csv("myfile.csv")
train, test = train_test_split(df)
chemprop_wrapper = ChemPropWrapper("logS")
pred = chemprop_wrapper.validate(train, test)
```

The **cross_validate** method in **run_benchmark.py** comes from the [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils).  The **cross_validate** function has four required arguments.

- **df** - a dataframe with a SMILES column  
- **model_list** - a list of tuples containing the model name and the wrapper class described above  
- **y_col** - the name of the column with the y value to be predicted  
- **group_list** - a list of group_names and group memberships (e.g. cluster ids), these can be calculated using the functions get_random_clusters, get_scaffold_clusters, get_butina_clusters, and get_umap_clusters in useful_rkdkit_utils.  

```python

y = "logS"
model_list = [("chemprop",ChemPropWrapper),("lgbm_morgan", LGBMMorganCountWrapper),("lgbm_prop",LGBMPropWrapper)]
group_list = [("random", uru.get_random_clusters),("butina", uru.get_butina_clusters)]
result_df = uru.cross_validate(df,model_list,y,group_list)
```



