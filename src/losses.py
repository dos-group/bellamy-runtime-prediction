import numpy as np
import torch
import torch.nn as nn
import copy
import pandas as pd
import logging
from ignite.engine import Engine

from src.utils import default_config


class TrainingLoss(object):
    def __init__(self, *args, **kwargs):
        
        self.__dict__.update(kwargs)
            
        self.prediction_loss = nn.SmoothL1Loss()
        self.reconstruction_loss = nn.MSELoss()
        
    def __call__(self, pred_dict: dict, batch):
        """Computes the loss given a batch-object and a result-dict from the model.
        
        Parameters
        ----------
        pred_dict : dict
            A result dict from a call to the model
        batch: Batch
            A batch object from the Dataloader

        Returns
        ------
        loss
            A PyTorch loss.
        """
        
        pred_dict = {k:v for k,v in pred_dict.items() if v is not None and "_codes" not in k}
                
        loss_list : list = []
        for k, v in pred_dict.items():
            ks = "_".join(k.split('_')[:-1])
            
            t = batch[ks]
            
            loss_value = None
            if ks == "y": # loss for runtime prediction
                loss_value = self.prediction_loss(v, t)
            else: # loss for auto-encoder reconstruction
                loss_value = self.reconstruction_loss(v, t)
                
            loss_list.append(loss_value)
        
        return sum(loss_list)

    
class FineTuningLoss(object):
    def __init__(self, *args, **kwargs):
        
        self.loss = nn.SmoothL1Loss()
        
    def __call__(self, pred_dict, batch):
        """Computes the loss given a batch-object and a result-dict from the model.
        
        Parameters
        ----------
        pred_dict : dict
            A result dict from a call to the model
        batch: Batch
            A batch object from the Dataloader

        Returns
        ------
        loss
            A PyTorch loss.
        """
        
        y_pred = pred_dict["y_pred"]
        y_true = batch.y
        return self.loss(y_pred, y_true)   
        

class ObservationLoss(object):
    def __init__(self, *args, **kwargs):

        self.loss = nn.L1Loss()

    def __call__(self, pred_dict, batch):

        y_pred = pred_dict["y_pred"]
        y_true = batch.y
        return self.loss(y_pred, y_true)        
        
        
class Scorer(object):
    def __init__(self, trainer: Engine, **kwargs):
        """
        Parameters
        ----------
        trainer : Engine
            The training-engine
        """
        
        self.key = None
        self.relation = None
        self.threshold = None
        self.trainer = trainer
        
        kwargs.setdefault('relation', 'lt')
        kwargs.setdefault('key', 'ft_loss')
        kwargs.setdefault('threshold', -5.)
                
        self.__dict__.update(kwargs)
        self.__dict__.update(default_config.get("score_function", {}))
        
        logging.info(f"Scorer-Configuration: {self.__dict__}")
        
    def __call__(self, engine: Engine):
        """Determines an improvement during training.
        
        Parameters
        ----------
        engine : Engine
            The training engine

        Returns
        ------
        score
            A score used for determining a training progress.
        """
        
        operator = min if self.relation == "lt" else max
        
        result = operator(self.threshold, -engine.state.metrics[self.key])
        
        if result == self.threshold:
            logging.info("EarlyStopping: Stop training")
            self.trainer.terminate()
        
        return result
    