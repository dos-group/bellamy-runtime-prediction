import os
import sys
import random
import logging
from datetime import datetime

import copy
from typing import Optional
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine, EarlyStopping, TerminateOnNan
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.contrib.handlers.param_scheduler import ConcatScheduler, LinearCyclicalScheduler, CosineAnnealingScheduler

from src.training_routines import create_supervised_trainer
from src.losses import *
from src.utils import default_config, update_flat_dicts

import time
from torch_geometric.nn.inits import glorot, zeros

    
class TorchModel(object):
    def __init__(self, path_to_dir: str, job_identifier: str):
        """
        Parameters
        ----------
        path_to_dir : str
            The path to a folder for temporary results
        job_identifier: str
            An identifier for the algorithm
        """
        
        self.__dict__.update(default_config.get("model_setup", {}))
        
        self.path_to_dir = path_to_dir
        self.job_identifier = job_identifier   
        
        self.model_class = NNModel
        self.optimizer_class = Adam
        self.training_loss_class = TrainingLoss
        self.fine_tuning_loss_class = FineTuningLoss

        
        self.model_args = {**self.__dict__.get("model_args", {}), "device": self.device}
        self.model_args.update({k:v for k,v in default_config.get("pipeline", {}).items() if "column" in k})
        
        self.optimizer_args = self.__dict__.get("optimizer_args", {})
        
        self.training_loss_args = {**self.__dict__.get("training_loss_args", {}), "device": self.device}
        self.fine_tuning_loss_args = {**self.__dict__.get("fine_tuning_loss_args", {}), "device": self.device}
        
        self.model: Optional[NNModel] = None
                 
    
    @property
    def config_dict(self):
        """Configuration dictionary."""
        return {
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            **self.model_args,
            **self.optimizer_args,
            **self.training_loss_args,
            **self.fine_tuning_loss_args
        }      
    
    def get_pretrained_model(self, checkpoint = None):
        """Get a model, optimizer and loss, optionally with pretrained states.
        
        Parameters
        ----------
        checkpoint : dict
            A checkpoint of a pretrained model (Optional)
        """
        
        config_dict = copy.deepcopy(self.config_dict)
        
        model_state_dict = None
        # load from checkpoint
        if checkpoint is not None:
            model_state_dict = checkpoint['model_state_dict']
            best_trial_config = checkpoint['best_trial_config']
            config_dict = {**config_dict, **best_trial_config, "device": self.device}
        
        ###################################################
        ### pre-training completed, if it was necessary ###
        ###################################################
        logging.info("Using config: {}".format(config_dict))
        
        ### extract and override ### 
           
        model_args, optimizer_args, fine_tuning_loss_args = update_flat_dicts(config_dict, [self.model_args, 
                                                                                            self.optimizer_args, 
                                                                                            self.fine_tuning_loss_args])
        ############################
        
        reuse_for_fine_tuning = default_config.get("reuse_for_fine_tuning", {})
        
        if not reuse_for_fine_tuning.get("model_args", False):
            model_args = self.model_args
            
        if not reuse_for_fine_tuning.get("optimizer_args", False):
            optimizer_args = self.optimizer_args
            
        ### create model ###
        model = self.model_class(**model_args).double()
        
        if model_state_dict is not None:
            # load pre-trained models and other utilities
            model.load_state_dict(model_state_dict)
            
        # freeze all layers
        model.freeze_all_layers()
                
        # disable dropout layers
        model.disable_dropout()
        
        if model_state_dict is None:
            # unfreeze scale out layer
            model.unfreeze_scale_out_layer()
            
        # unfreeze final prediction layer
        model.unfreeze_c_layer()
        
            
        model = model.to(torch.double).to(self.device)
        
        logging.info(f"#Parameters: {model.all_params}")
        logging.info(f"Trainable #parameters: {model.all_trainable_params}")

        # init optimizer and loss
        optimizer = self.optimizer_class(model.parameters(), **optimizer_args)
        loss = self.fine_tuning_loss_class(**fine_tuning_loss_args)
        
        suffix = "PreTrained-" if checkpoint is not None else ""
        logging.info(f"{suffix}Model: {self.model_class}, {suffix}Args={model_args}")
        logging.info(f"{suffix}Optimizer: {self.optimizer_class}, {suffix}Args={optimizer_args}")
        logging.info(f"{suffix}Loss: {self.fine_tuning_loss_class}, {suffix}Args={fine_tuning_loss_args}")
        
        return model, optimizer, loss, config_dict
    
    
    def predict(self, dataset: list):
        """Predict the runtime of job(s) given certain configuration(s).
        
        Parameters
        ----------
        dataset : list
            A list of Data-objects
        
        Returns
        ----------
        list
            a list of prediction results
        """
        
        result_list = []
        
        target_keys : list = ["y_pred"]
        
        self.model.eval()
        with torch.no_grad():
            for b in DataLoader(dataset, batch_size=self.batch_size, follow_batch=self.follow_batch):
                b = b.to(self.device)
                res_dict : dict = self.model(b)
                result_list += [{k:v for k,v in res_dict.items() if k in target_keys}]
                
        return result_list   
     
    
    def fit(self, dataset:list, checkpoint = None):
        """Fit / Fine-tune a model.
        
        Parameters
        ----------
        dataset : list
            A list of Data-objects
        checkpoint : dict
            A checkpoint of a pretrained model (Optional)
        """
                            
        logging.info(f"dataset: {len(dataset)}")
        
        # get pretrained model, if any
        model, optimizer, loss, config_dict = self.get_pretrained_model(checkpoint=checkpoint)

        if len(dataset):
            
            # setup ignite trainer
            trainer = create_supervised_trainer(
                model, 
                optimizer, 
                loss_fn=loss, 
                device=self.device, 
                non_blocking=True,
                output_transform=lambda x, y, y_pred, loss: (y_pred, y)
            )
                        
            to_save: dict = {
                "model_state_dict": model,
                "optimizer_state_dict": optimizer,
                "trainer_state_dict": trainer
            }
            
            # setup ignite disksaver
            save_handler = DiskSaver(self.path_to_dir, 
                                     create_dir=True,
                                     require_empty=False, 
                                     atomic=True)
            
            filename_prefix : str = f"{self.job_identifier}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_noice{random.randint(0, 10000):04d}"
            
            # scorer function to determine training improvements
            score_function = Scorer(trainer)    
            Loss(ObservationLoss()).attach(trainer, "ft_loss")
            
            # setup ignite checkpoint handler
            checkpoint_handler = Checkpoint(
                    to_save,
                    save_handler,
                    filename_prefix=filename_prefix,
                    score_function=score_function,
                    global_step_transform=global_step_from_engine(trainer),
                    n_saved=1)
            
            # configure early stopping to counter early stopping
            stopping_handler = EarlyStopping(score_function=score_function, 
                                            trainer=trainer, 
                                            patience=default_config.get("early_stopping", {}).get("patience", 100))
            
            
            trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, stopping_handler)
            
            # setup cyclical annealing of learning rate
            logging.info("Use Concat-Scheduler!")
            scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.001, end_value=0.01, cycle_size=20)
            scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.01, end_value=0.001, cycle_size=40)

            combined_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=[10, ])
            trainer.add_event_handler(Events.EPOCH_STARTED, combined_scheduler)
                
            
            batch_size = config_dict.get('batch_size', self.batch_size)
            epochs = self.epochs[1]
            
            # data loader for loading batches of training data
            train_loader = DataLoader(dataset, 
                                      shuffle=True, 
                                      batch_size=batch_size, 
                                      follow_batch=self.follow_batch)
            
            @trainer.on(Events.EPOCH_COMPLETED)
            def maybe_enable_scale_out_modeling(trainer):
                if len(dataset) > 1 and trainer.state.epoch == int(7.5 * (2**len(dataset))):
                    model.unfreeze_scale_out_layer()
                      
            # start training        
            trainer.run(train_loader, max_epochs=epochs)
            
            # after fine-tuning, load best model state and cache model
            best_checkpoint_path = os.path.join(self.path_to_dir, checkpoint_handler.last_checkpoint)
            checkpoint = torch.load(best_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        self.model = model
            
        return self


##################################################################
################ Actual PyTorch model below ######################
##################################################################
def init_weights(m):
    """He Initialization."""
    if type(m) == nn.Linear:
        # weights are using lecun-normal initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='selu')
        # biases zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        

class NNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NNModel, self).__init__()
        
        self.__dict__.update(kwargs)
        
        self.downscale_hidden_dim : int = int(self.hidden_dim / 2)
        self.upscale_hidden_dim : int = int(self.hidden_dim * 2)
          
        ### instance count embeddings ###
        self.scale_out_layer = nn.Sequential(nn.Linear(3, self.upscale_hidden_dim),
                                             nn.SELU(),
                                             nn.Linear(self.upscale_hidden_dim, self.hidden_dim), 
                                             nn.SELU())
        self.scale_out_layer.apply(init_weights)
        
        ### encoder ###
        self.encoder = nn.Sequential(nn.Linear(self.encoding_dim, self.hidden_dim, bias=False),
                                     nn.SELU(),
                                     nn.AlphaDropout(p=self.dropout),
                                     nn.Linear(self.hidden_dim, self.downscale_hidden_dim, bias=False), 
                                     nn.SELU())
        self.encoder.apply(init_weights)
        
        ### decoder ###
        self.decoder = nn.Sequential(nn.Linear(self.downscale_hidden_dim, self.hidden_dim, bias=False),
                                     nn.SELU(),
                                     nn.AlphaDropout(p=self.dropout),
                                     nn.Linear(self.hidden_dim, self.encoding_dim, bias=False), 
                                     nn.Tanh())
        self.decoder.apply(init_weights)
        
        
        ### combine predictions ###
        # in_dim = output dim of scale out layer + number of encoded columns + one more for additional
        self.c_layer_in_dim = self.hidden_dim + int((len(self.emb_columns) + 1) * self.downscale_hidden_dim)
        
        self.c_layer = nn.Sequential(nn.Linear(self.c_layer_in_dim, self.hidden_dim), 
                                     nn.SELU(),
                                     nn.Linear(self.hidden_dim, 1), 
                                     nn.SELU())
        self.c_layer.apply(init_weights)
        
    
    def freeze_all_layers(self):
        logging.info("Freeze all layers...")
        
        for param in self.parameters():
            param.requires_grad = False
    
    
    def disable_dropout(self):
        logging.info("Disable dropout layers...")
        
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.AlphaDropout)):
                module.p = 0.0
    
    
    def unfreeze_scale_out_layer(self):
        logging.info("Unfreeze Scale-Out-Layer...")
        
        for param in self.scale_out_layer.parameters():
            param.requires_grad = True
        
    
    def unfreeze_c_layer(self):
        logging.info("Unfreeze C-Layer...")
        
        for param in self.c_layer.parameters():
            param.requires_grad = True

    
    @property
    def config_dict(self):
        return {
            "device": self.device,
            "encoding_dim": self.encoding_dim,
            "hidden_dim": self.hidden_dim,
            "downscale_hidden_dim": self.downscale_hidden_dim,
            "upscale_hidden_dim": self.upscale_hidden_dim,
            "dropout": self.dropout,
            "c_layer_in_dim": self.c_layer_in_dim
        }
    
    @property
    def all_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @property
    def all_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, batch):
        
        ### instance count embeddings ###
        instance_count = torch.cat([batch.instance_count_div, batch.instance_count_log, batch.instance_count], dim=-1)
        instance_count = self.scale_out_layer(instance_count)
        
        ### compute required embeddings ###
        x_emb_enc = self.encoder(batch.x_emb)
        
        emb_codes = x_emb_enc.detach().clone()
        
        x_emb_dec = self.decoder(x_emb_enc)
        
        emb_pred_dense, _ = to_dense_batch(x_emb_enc, batch.x_emb_batch)
        emb_pred = emb_pred_dense.reshape(emb_pred_dense.size(0), -1)
        
        ### compute optional embeddings ###
        x_opt_dec = None
        opt_codes = None
        if len(batch.x_opt):
            x_opt_enc = self.encoder(batch.x_opt)
            
            opt_codes = x_opt_enc.detach().clone()
            
            x_opt_dec = self.decoder(x_opt_enc)
            
            opt_pred = global_mean_pool(x_opt_enc, batch.x_opt_batch)
        else:
            opt_pred = torch.zeros(emb_pred_dense.size(0), emb_pred_dense.size(2), device=self.device, dtype=x_emb_dec.dtype)
        
        ### combine predictions ###
        y_pred = self.c_layer(torch.cat([instance_count, emb_pred, opt_pred], dim=-1))
                
        return {
            "y_pred": y_pred,
            "x_emb_dec": x_emb_dec,
            "x_opt_dec": x_opt_dec,
            "emb_codes": emb_codes,
            "opt_codes": opt_codes
        }
