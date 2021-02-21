import numpy as np
import pandas as pd
import logging
import os
import copy
import math
import dill
import torch
from itertools import product
from src.utils import default_config
from src.pipeline import TorchPipeline
from src.model import TorchModel
from src.preparation import load_data


class BellamyModel(object):
    def __init__(self, training_target: str,  **kwargs): 
        """
        Parameters
        ----------
        training_target : str
            Decide which data to use for pretraining
        """
        
        self.training_target = training_target
        self.device = torch.device("cpu")
        
        self.data_path = default_config.get("path_to_data_dir")
        
        self.root_path = os.path.join(default_config.get("path_to_pretrain_dir"), self.training_target.lower())
        self.temp_path = os.path.join(default_config.get("path_to_temp_dir"), self.training_target.lower())
        

    def __load_from_disk__(self, job_type: str):
        """Load the most recent pipeline and pretraining-checkpoint.
        
        Parameters
        ----------
        job_type : str
            The scalable analytics algorithm
        """
        
        self.model_wrapper = TorchModel(self.temp_path, job_type)
        self.pipeline = TorchPipeline.getInstance(self.root_path, job_type)
        self.checkpoint = None
        if os.path.exists(os.path.join(self.root_path, f"{job_type}_checkpoint.pt")):
            self.checkpoint = torch.load(os.path.join(self.root_path, f"{job_type}_checkpoint.pt"), map_location=self.device)
                            

    def __call__(self, payload: dict):
        """Find best scale-out out of a range according to the smallest runtime for a specific configuration.
        
        Parameters
        ----------
        payload : dict
            A submitted payload
        
        Returns
        ----------
        dict
            The response dict
        """
        
        job_type = payload["job_type"]
        self.__load_from_disk__(job_type)
        
        if not self.pipeline.is_fitted:
            return {"response": "Pipeline is not yet pre-trained."}
                
        if self.checkpoint is None:
            return {"response": "Pre-trained checkpoint does not yet exist."}
        
        ### unpack request object ###
        min_scale_out = payload["min_scale_out"]
        max_scale_out = payload["max_scale_out"]        
        
        essential_properties = payload["essential_properties"]
        optional_properties = payload.get("optional_properties", {})
        #############################
        
        # check if all necessary properties are included
        for prop in default_config.get("pipeline", {}).get("emb_columns", []):
            if prop not in essential_properties:
                raise ValueError(f"Missing essential property '{prop}'!")
        
        # fill up optional properties if they were not provided
        for prop in default_config.get("pipeline", {}).get("opt_columns", []):
            if prop not in optional_properties:
                optional_properties[prop] = ""
        
        ### fine-tuning, if possible ###
        fitted : bool = False
        
        historical_data = load_data(self.data_path, self.training_target)
        if len(historical_data) and job_type in historical_data:
            historical_job_data = historical_data[job_type]
            for k, v in essential_properties.items():
                if len(historical_job_data) and bool(v):
                    historical_job_data = historical_job_data.loc[historical_job_data[k] == v, :]
            
            if len(historical_job_data):
                train_list = self.pipeline.transform(historical_job_data)
                if len(train_list):
                    self.model_wrapper.fit(train_list, checkpoint = self.checkpoint)
                    fitted = True
        
        if not fitted:
            self.model_wrapper.fit([], checkpoint = self.checkpoint)
        ################################
        
        ### (prepare) inference ###
        all_scale_outs = list(range(min_scale_out, max_scale_out + 1))
        pred_df = {k:([v] * len(all_scale_outs)) for k,v in dict(**essential_properties, **optional_properties).items()}
        pred_df = pd.DataFrame.from_dict(pred_df)
        
        tol : float = 0.000001
        pred_df.loc[:, 'instance_count'] = pd.Series(np.array(all_scale_outs), index=pred_df.index)
        pred_df.loc[:, 'instance_count_log'] = pred_df['instance_count'].map(lambda ic: math.log(float(ic)))
        pred_df.loc[:, 'instance_count_div'] = pred_df['instance_count'].map(lambda ic: 1. / (float(ic) + tol))
        
        pred_df.loc[:, 'type'] = pred_df['machine_type'].map(lambda mt: 0)
        pred_df.loc[:, 'gross_runtime'] = pred_df['machine_type'].map(lambda mt: -1)
        
        pred_list = self.pipeline.transform(copy.deepcopy(pred_df))
        pred_list = self.model_wrapper.predict(pred_list)
        pred_list = [el["y_pred"].tolist() for el in pred_list]
        pred_list = sum(pred_list, [])
        
        pred_df.loc[:, 'gross_runtime'] = pred_list
        #########################
        # get best scale-out according to smallest predicted runtime
        min_element = pred_list.index(min(pred_list))
        best_row = pred_df.iloc[min_element, :]
        
        best_setup = f"scale-out={best_row['instance_count']}"
                
        return {
            "best": f"Fastest setup is '{best_setup}', with {pred_list[min_element][-1]:.2f}s.",
            "complete": [f"scale-out={row['instance_count']} with {row['gross_runtime']}s" for _, row in pred_df.iterrows()]
        }
        
        