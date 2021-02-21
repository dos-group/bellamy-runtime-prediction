import numpy as np
import torch
import torch.nn as nn
import copy
import pandas as pd
import logging
from ignite.engine import Engine

from src.utils import default_config


class LossWrapper(object):
    def __init__(self, *args, **kwargs):
        
        self.__dict__.update(kwargs)
            
        self.prediction_loss = nn.SmoothL1Loss(reduction="none")
        self.reconstruction_loss = nn.MSELoss(reduction="none")
        
        self.train_embeddings = True
        
        # default: both "classes" have same weight
        self.weights = torch.tensor([1., 1.]).to(self.device).double()
        
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
        
        types = torch.abs(batch.type.to(torch.long))
        
        loss_list : list = []
        for k, v in pred_dict.items():
            ks = "_".join(k.split('_')[:-1])
            
            t = batch[ks]
            
            loss_values = None
            if ks == "y": # loss for runtime prediction
                loss_values = self.prediction_loss(v, t)
            elif self.train_embeddings: # loss for auto-encoder reconstruction
                loss_values = self.reconstruction_loss(v, t)
            else:
                continue
                        
            loss_values = torch.mean(loss_values, dim=-1).reshape(-1)
            
            weights = self.weights[types].reshape(-1, 1)
            weights = weights.repeat(1, int(len(t) / len(batch.y))).reshape(-1)
            
            loss_value = torch.sum(weights * loss_values) / torch.sum(weights) 
                
            loss_list.append(loss_value)
        
        return sum(loss_list)

    
class FineTuningLoss(object):
    def __init__(self, threshold: float, **kwargs):
        """
        Parameters
        ----------
        threshold : float
            A threshold for early stopping
        """
        
        if not isinstance(threshold, (int, float, str)):
            raise ValueError("threshold must be value.")
        
        self.loss = nn.SmoothL1Loss()
        self.threshold = abs(float(threshold)) * 1.0
        
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
                
        types = torch.abs(batch.type.to(torch.long))
        
        if torch.sum(types == 0) > 0:
            y_pred = pred_dict["y_pred"]
            y_true = batch.y
            return self.loss(y_pred[types == 0], y_true[types == 0])   
        else:
            return torch.tensor(self.threshold)
        
        
class Scorer(object):
    def __init__(self, trainer: Engine, **kwargs):
        """
        Parameters
        ----------
        trainer : Engine
            The training-engine
        """
        
        kwargs.setdefault('relation', 'lt')
        kwargs.setdefault('key', 'ft_loss')
        kwargs.setdefault('threshold', -5.)
        
        self.trainer = trainer
        
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
    