import os
import dill
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MinMaxScaler
from src.transforms import *
from src.utils import create_dirs, default_config
    

transformer_dict : dict = {
    "BinaryTransformer": BinaryTransformer,
    "HashingVectorizer": HashingVectorizer,
    "MinMaxScaler": MinMaxScaler
}

    
class TorchPipeline():
    def __init__(self, path_to_dir: str, algorithm: str, **kwargs):
        """
        Parameters
        ----------
        path_to_dir : str
            The path to the pipeline-object
        algorithm: str
            The scalable analysis algorithm
        """
        
        self.path_to_dir = path_to_dir
        self.algorithm = algorithm
        self.is_fitted = False
        create_dirs(self.path_to_dir)
                
        self.pipeline = self._get_pipeline()
        
    @classmethod
    def getInstance(cls, path_to_dir: str, algorithm: str, **kwargs):
        """Retrieves a machine learning pipeline. Creates it if it does not exist yet.

        Parameters
        ----------
        path_to_dir : str
            The path to the pipeline-object
        algorithm: str
            The scalable analysis algorithm

        Returns
        ------
        Pipeline
            The machine learning pipeline
        """
        
        path_to_file = os.path.join(path_to_dir, f"{algorithm}_pipeline.pkl")
        
        if os.path.exists(path_to_file):
            logging.info(f"Load pretrained pipeline for {algorithm}...")
            with open(path_to_file, 'rb') as dill_file:
                return dill.load(dill_file)  
        else:
            return TorchPipeline(path_to_dir, algorithm, **kwargs)  
        
    def save(self):
        """Save the pipeline to disk."""
        with open(os.path.join(self.path_to_dir, f"{self.algorithm}_pipeline.pkl"), "wb") as dill_file:
            self.is_fitted = True
            dill.dump(self, dill_file)
    
    def fit(self, *args, **kwargs):
        return self.pipeline.fit(*args, **kwargs)
    
    def transform(self, *args, **kwargs):
        return self.pipeline._transform(*args, **kwargs)
              
    def _get_pipeline(self):
        """Retrieves a machine learning pipeline.

        Returns
        ------
        Pipeline
            The machine learning pipeline.
        """
            
        transformer_list : list = []
        pipeline_config : dict = default_config.get("pipeline", {})
        for idx, t in enumerate(pipeline_config.get("transforms", [])):
            transformer_list.append((f"feature_transform{idx}", 
                                        WrapperTransformer(in_column=t.get("in_column"), 
                                                        out_column=t.get("out_column"), 
                                                        transformer=transformer_dict.get(t.get("transformer"))(**t.get("transformer_args", {}))
                                                        )))
        
        transformer_list.append(('to_graph_list', ToGraphListTransformer(**pipeline_config)))
        
        return Pipeline(transformer_list)
        
