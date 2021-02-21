import pandas as pd
import numpy as np
import os
import math
import copy
import logging
from collections import OrderedDict

col_transforms = {
    "Runtime": lambda x: x / 1000, # ms to s
}

separators = {
    ".tsv": "\t",
    ".csv": ","
}

header_transforms = {
    "Nodes": "instance_count",
    "Runtime": "gross_runtime"
}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def transform_bell_df(df: pd.DataFrame, algorithm: str):
    """Prepares a Bell-DataFrame.

    Parameters
    ----------
    df : DataFrame
        The raw DataFrame
    algorithm: str
        Name of the algorithm

    Returns
    ------
    DataFrame
        the adapted DataFrame.
    """
        
    df.loc[:, 'machine_type'] = df['instance_count'].map(lambda mt: "wallynode")
    df.loc[:, 'job_type'] = df['machine_type'].map(lambda mt: f"{algorithm}-spark")
    
    df.loc[:, 'slots'] = df['machine_type'].map(lambda mt: 4) * df['instance_count'].astype('int64')
    df.loc[:, 'memory'] = df['machine_type'].map(lambda mt: 16000) * df['instance_count'].astype('int64')
    
    df.loc[:, 'data_characteristics'] = df['machine_type'].map(lambda mt: "") # data characteristics: N.A.
    
    df.loc[:, 'environment'] = df['machine_type'].map(lambda mt: "private cluster yarn") 
            
    if algorithm == "grep":
        df.loc[:, 'data_size_MB'] = df['machine_type'].map(lambda mt: 1000.0) * 250 # 250 GB
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "berlin") # parameters: "berlin" (filter-word)
        
    elif algorithm == "pagerank":
        df.loc[:, 'data_size_MB'] = df['machine_type'].map(lambda mt: 1000.0) * 3.4 # 3.4 GB
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "5") # parameters: 5 iterations
        
    elif algorithm == "sgd":
        df.loc[:, 'data_size_MB'] = df['machine_type'].map(lambda mt: 1000.0) * 10 # 10 GB
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "100 1.0") # parameters: 100 iterations, step_size = 1.0
        
    return df


def transform_c3o_df(df: pd.DataFrame, algorithm: str):  
    """Prepares a C3O-DataFrame.

    Parameters
    ----------
    df : DataFrame
        The raw DataFrame
    algorithm: str
        Name of the algorithm

    Returns
    ------
    DataFrame
        the adapted DataFrame.
    """
        
    df.loc[:, 'job_type'] = df['machine_type'].map(lambda mt: f"{algorithm}-spark")
    
    df.loc[:, 'environment'] = df['machine_type'].map(lambda mt: "public cloud aws emr") 
        
    if algorithm == "grep":
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "computer") # parameters: "computer" (filter-word)
        df.loc[:, 'data_characteristics'] = df[['line_length', 'lines', 'p_occurrence']].apply(lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
        
    elif algorithm == "kmeans":
        df.loc[:, 'job_args'] = df['k'].map(lambda k: f"{k} 0.001") # parameters: values for k + fixed convergence criterion
        df.loc[:, 'data_characteristics'] = df[['features', 'observations']].apply(lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
        
    elif algorithm == "pagerank":
        df.loc[:, 'job_args'] = df['convergence_criterion'].map(lambda cc: cc) # parameters: values for convergence criterion
        df.loc[:, 'data_characteristics'] = df[['pages', 'links']].apply(lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
        
    elif algorithm == "sgd":
        df.loc[:, 'job_args'] = df['iterations'].map(lambda i: i) # parameters: values for iteration
        df.loc[:, 'data_characteristics'] = df[['features', 'observations']].apply(lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
        
    elif algorithm == "sort":
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "") # parameters: None
        df.loc[:, 'data_characteristics'] = df[['line_length', 'lines']].apply(lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
        
    return df


def remove_outlier_runs(df: pd.DataFrame):
    """Remove outliers in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The raw DataFrame

    Returns
    ------
    DataFrame
        the adapted DataFrame.
    """
    
    values = df["instance_count"].values.flatten()
    llist = []
    for v in np.unique(values):
        sub_df = df.loc[values == v, "gross_runtime"]
        indices = np.argsort(sub_df.values)[1:-1]
        llist.append(df.iloc[sub_df.index[indices], :])
    return pd.concat(llist, ignore_index=True)


df_transforms = {
    "bell": transform_bell_df,
    "c3o": transform_c3o_df
}


def load_data(root_dir: str, training_target: str, **kwargs):
    """Loads data from disk.

    Parameters
    ----------
    root_dir : str
        The root directory of the data
    training_target: str
        What data to incorporate

    Returns
    ------
    dict
        A dict of DataFrame's.
    """
    
    res_dict: OrderedDict = OrderedDict()
    
    sub_dirs = [training_target.lower()] if training_target != "all" else ["bell", "c3o"]
        
    for sub_dir in sub_dirs:
    
        path_to_dir : str = os.path.join(root_dir, sub_dir)
        for el in sorted(os.listdir(path_to_dir)):
            if ".csv" in el or ".tsv" in el:
                algorithm = ".".join(el.split(".")[:-1])
                ext = os.path.splitext(el)[1]
                sep = separators.get(ext)
                
                # load data
                df = pd.read_csv(os.path.join(path_to_dir, el), sep=sep, **kwargs)
                
                # apply simple column transformations
                for col in df.columns:
                    df[col] = df[col].apply(col_transforms.get(col, lambda x: x))
                
                # rename columns if needed
                df.columns = [header_transforms.get(col, col) for col in list(df.columns)]
                
                # further datatrame transformations if needed
                df = df_transforms.get(sub_dir, lambda x,y: x)(df, algorithm)
                
                job = el.replace(f"{ext}", "")
                
                # remove outlier runs
                if sub_dir == "bell":
                    logging.info(f"[{sub_dir}] Remove outlier runs...")
                    df = remove_outlier_runs(df)
                
                # reset index
                df = df.reset_index(drop=True)
                
                tol : float = 0.000001
                df.loc[:, 'instance_count_log'] = df['instance_count'].map(lambda ic: math.log(float(ic)))
                df.loc[:, 'instance_count_div'] = df['instance_count'].map(lambda ic: 1. / (float(ic) + tol))
                
                df.loc[:, 'type'] = df['machine_type'].map(lambda mt: 0)
                
                if job in res_dict:
                    res_dict[job].append(df)
                else:
                    res_dict[job] = [df]
    
    for key in list(res_dict.keys()):
        res_dict[key] = pd.concat(res_dict[key], ignore_index=True)
    
    return res_dict
    