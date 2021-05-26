import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import logging
from typing import List, Union


class PairData(Data):
    def __inc__(self, key, value):
        """Define how batching shall work. Refer to: https://pytorch-geometric.readthedocs.io/en/1.6.3/notes/batching.html#pairs-of-graphs"""
        if key == 'edge_index_emb':
            return self.x_emb.size(0)
        if key == 'edge_index_opt':
            return self.x_opt.size(0)
        else:
            return super(PairData, self).__inc__(key, value)
    

class ToGraphListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, emb_columns:list=[], opt_columns:list=[], extra_columns:list=[], target_column:list=["y"], device="cpu", **kwargs):
        """
        Parameters
        ----------
        emb_columns : list
            The columns for the essential context properties
        opt_columns : list
            The columns for the optional context properties
        extra_columns: list
            A list of properties that will be additionally attached to the Data-object
        target_column: list
            The column to use as target during training
        devide: str
            The device available for training
        """
        
        super().__init__()
        
        self.emb_columns = emb_columns or []
        self.opt_columns = opt_columns or []
        
        self.extra_columns = extra_columns
        self.target_column = target_column
        
        self.device = device
        self.conn_dict : dict = {}
    
    @staticmethod
    def get_values(row: pd.Series, cols):  
        """Retrieve values from Pandas-Series.

        Parameters
        ----------
        row : Series
            An input Series
        cols: list
            A list of columns

        Returns
        ------
        Tensor
            A torch-Tensor.
        """ 
        
        if not len(cols): return torch.tensor([])
             
        col_values = row[cols]
        value_list = col_values.values.tolist()
        value_array = np.array([sum(el, []) if isinstance(el, list) and all([isinstance(l, list) for l in el]) else el for el in value_list])
        value_array = value_array.astype(np.float64)
        return torch.from_numpy(value_array).reshape(len(cols), -1)
    
    @staticmethod
    def create_edge_index(conn_dict:dict, num_nodes:int):
        """Create an edge-index for a graph.

        Parameters
        ----------
        conn_dict : dict
            A cached dictionary
        num_nodes: int
            Number of nodes in the graph

        Returns
        ------
        dict
            The updated cached dictionary.
        Tensor
            A tensor describing the graph connectivity.
        """
        
        if num_nodes == 0: return conn_dict, torch.tensor([])
        
        if num_nodes not in conn_dict:
            conn_dict[num_nodes] = torch.ones(num_nodes, num_nodes).nonzero(as_tuple=False).t()
        
        return conn_dict, conn_dict[num_nodes]
        
    def fit(self, X, y = None):
        """ For the purpose of compatibility."""
        return self
    
    def predict(self, *args, **kwargs):
        """ For the purpose of compatibility."""
        return self.transform(*args, **kwargs)
    
    def transform(self, X, y = None):
        """Transform a DataFrame to a list.

        Parameters
        ----------
        X : DataFrame
            An input DataFrame

        Returns
        ------
        list
            A list of PyTorch-Data objects.
        """
        
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Expected 'X' to be a pandas DataFrame or Series.")
                        
        result_list = []
        for _, row in X.iterrows():
            # extract features / x (add nodes and their features)
            props : dict = {
                "x_emb": self.get_values(row, self.emb_columns).to(self.device),
                "x_opt": self.get_values(row, self.opt_columns).to(self.device)
            }
            # get addititional properties
            for col in self.extra_columns:
                props[col] = self.get_values(row, [col]).to(self.device)   
            # extract label
            props["y"] = self.get_values(row, self.target_column).to(self.device)
            
            # create and add edge_index (describes connectivity of graph)
            self.conn_dict, props["edge_index_emb"] = self.create_edge_index(self.conn_dict, props["x_emb"].size(0))
            self.conn_dict, props["edge_index_opt"] = self.create_edge_index(self.conn_dict, props["x_opt"].size(0))
            
            # add to list
            result_list.append(PairData(**props))
                
        return result_list
    
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns:list=[]):
        """
        Parameters
        ----------
        columns : list
            the target columns
        """
        
        super().__init__()
        self.columns = columns
        
    def fit(self, X, y = None):
        """ For the purpose of compatibility."""
        return self
    
        
    def transform(self, X, y = None):
        """Select only few columns of DataFrame.

        Parameters
        ----------
        X : DataFrame
            An input DataFrame

        Returns
        ------
        DataFrame
            The filtered DataFrame.
        """
        
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Expected 'X' to be a pandas DataFrame or Series.")
        
        intersect_cols = list(set(self.columns).intersection(list(X.columns)))
        if intersect_cols:
            X = X.loc[:, intersect_cols]
            
        return X
    
    def __str__(self):
        return f'{self.__class__.__name__}(columns={self.columns})'
    
    def __repr__(self):
        return f'{self.__class__.__name__}(columns={self.columns})'
    

class RowSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column:str="", value=None):
        """
        Parameters
        ----------
        column : str
            the target column
        value: [str, int]
            the value to use for selection
        """
        
        super().__init__()
        self.column = column
        self.value = value
        
    def fit(self, X, y = None):
        """ For the purpose of compatibility."""
        return self
    
        
    def transform(self, X, y = None):
        """Filter a DataFrame based on a condition.

        Parameters
        ----------
        X : DataFrame
            An input DataFrame

        Returns
        ------
        DataFrame
            A filtered DataFrame.
        """
        
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Expected 'X' to be a pandas DataFrame or Series.")
        
        if self.column in list(X.columns):
            X = X.loc[X[self.column] == self.value, :]
            
        return X
    
    def __str__(self):
        return f'{self.__class__.__name__}(column={self.column}, value={self.value})'
    
    def __repr__(self):
        return f'{self.__class__.__name__}(column={self.column}, value={self.value})'
    
    
class WrapperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, in_column:Union[str,list]=None, out_column:[str,list]=None, transformer=None):
        """
        Parameters
        ----------
        in_column : [str, list]
            the target column(s)
        out_column: [str, list]
            the output column(s)
        transformer: func
            The transformer to use
        """
        
        super().__init__()
        self.in_column = in_column if isinstance(in_column, list) else [in_column]
        self.out_column = out_column if isinstance(out_column, list) else [out_column]
        self.transformer = transformer
        
        self.vocabulary = list("abcdefghijklmnopqrstuvwxyz0123456789.:,;_-+#@")
        
    def fit(self, X, y=None):
        """Fits an internal transformer.

        Parameters
        ----------
        X : DataFrame
            An input DataFrame
        """
        
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Expected 'X' to be a pandas DataFrame or Series.")
            
        if isinstance(self.transformer, MinMaxScaler):
            unique_types = np.unique(X["type"].values)
            
            if len(unique_types) == 1: # not pre-trained / does not matter
                self.transformer.fit(X[self.in_column].values.reshape(-1, len(self.in_column)))
            else: # pre-trained because of support rows
                self.transformer.fit(X.loc[X.type == 1, self.in_column].values.reshape(-1, len(self.in_column)))
                
        return self

    def __transform_with_hasher__(self, row: pd.Series):
        values: List[List[Union[int, float]]] = []
        for el in row.to_list():
            el_char_list = list(str(el))
            # remove undesired characters
            el_char_list = [c for c in el_char_list if c in self.vocabulary]
            el_str = "".join(el_char_list)
            # get encoding from hashing vectorizer
            base_encoding: List[Union[int, float]] = self.transformer.transform([el_str]).toarray().reshape(-1).tolist()
            values.append(base_encoding)

        values = [[0] + sub_list for sub_list in values]  # category "hasher"
        return pd.Series(values)

    def __transform_with_binarizer__(self, row: pd.Series):
        values: List[List[Union[int, float]]] = [self.transformer.transform([e]).flatten().tolist()
                                                 for e in row.to_list()]
        values = [[1] + sub_list for sub_list in values]  # category "binarizer"
        return pd.Series(values)
    
    def transform(self, X, y = None):
        """Transforms certain DataFrame-columns using an internal transformer.

        Parameters
        ----------
        X : DataFrame
            An input DataFrame

        Returns
        ------
        DataFrame
            A DataFrame with modified columns.
        """
        
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Expected 'X' to be a pandas DataFrame or Series.")
            
        is_subset:bool = set(self.in_column).issubset(set(list(X.columns)))
                    
        if isinstance(self.transformer, HashingVectorizer) and is_subset:
            X[self.out_column] = X.loc[:, self.in_column].apply(self.__transform_with_hasher__, axis=1)
            
        elif isinstance(self.transformer, BinaryTransformer) and is_subset:
            X[self.out_column] = X.loc[:, self.in_column].apply(self.__transform_with_binarizer__, axis=1)
        
        elif is_subset:
            X[self.out_column] = self.transformer.transform(X[self.in_column].values.reshape(-1, len(self.in_column)))
        
        # just for some verbose action
        if isinstance(self.transformer, MinMaxScaler):
            logging.info(f"MinMaxScaler: data_min_={self.transformer.data_min_}, data_max_={self.transformer.data_max_}")
        
        return X
    

class BinaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n:int=40):
        """
        Parameters
        ----------
        n : int
            Size of result vector (default is 40)
        """
        
        super().__init__()
        self.n = n
        
        
    def fit(self, X, y=None):
        """ For the purpose of compatibility."""
        return self
    
    
    def _get_bin(self, x):
        """Converts a value to a binary representation as list.

        Parameters
        ----------
        x : [int, str, list]
            value that needs to be converted

        Returns
        ------
        list
            List of values, a binary representation.
        """ 
        x = 0 if (isinstance(x, str) and len(x) == 0) else int(x)
        return list(map(int, list(format(x, 'b').zfill(self.n))[-self.n:]))
    
    
    def transform(self, X, y = None):
        """Converts a list of values.

        Parameters
        ----------
        X : list
            A list of values that needs to be converted

        Returns
        ------
        array
            vector representations of numbers.
        """     
        return np.array([self._get_bin(val) for val in X])