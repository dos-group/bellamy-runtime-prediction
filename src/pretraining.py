import os
import dill
import torch
import logging
import time
import copy
import numpy as np
from functools import partial
import torch.nn as nn
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter, ExperimentAnalysis
from ignite.engine import Events
from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.training_routines import create_supervised_trainer, create_supervised_evaluator
from src.utils import default_config, update_flat_dicts, create_dirs
from src.model import TorchModel
from src.pipeline import TorchPipeline
from src.preparation import load_data


class Pretrainer(object):
    
    def __init__(self, training_target: str, **kwargs):
        """
        Parameters
        ----------
        training_target : str
            decide what data to use for pretraining
        """
        
        self.training_target = training_target
        self.root_path = os.path.join(default_config.get("path_to_pretrain_dir"), self.training_target.lower())
        self.data_path = default_config.get("path_to_data_dir")
        
    def __call__(self, job_type: str):
        """Gets called whenever a pretraining is required.
        
        Parameters
        ----------
        job_type : str
            The scalable analytics algorithm
        """
        
        historical_data = load_data(self.data_path, self.training_target)        
        dataset = historical_data.get(job_type, None)
        
        if dataset is None:
            logging.info("No data for pretraining available.")
            return
        
        pipeline = TorchPipeline.getInstance(self.root_path, job_type)
        data_list = pipeline.fit(copy.deepcopy(dataset)).transform(dataset)
        
        model = TorchModel(self.root_path, job_type)
        self.__dict__.update(**model.__dict__)
        
        hptuner_instance = HPTuner(**self.__dict__)
        
        result_dict: dict = HPTuner.perform_tuning(
            hptuner_instance,
            self.epochs[0],
            data_list)

        target_path = os.path.join(self.root_path, f"{job_type}_checkpoint.pt")   
        
        torch.save(result_dict, target_path)
        
        pipeline.save()


class HPTuner(object):
    def __init__(self, *args, **kwargs):
        
        self.__dict__.update(kwargs)
                
        logging.info(f"Default-Model: {self.model_class}, Default-Args={self.model_args}")
        logging.info(f"Default-Optimizer: {self.optimizer_class}, Default-Args={self.optimizer_args}")
        logging.info(f"Default-Loss: {self.training_loss_class}, Default-Args={self.training_loss_args}")
        logging.info(f"Default-Config: Device={self.device}, BatchSize={self.batch_size}")
        
        
    @staticmethod
    def compute_groups(dataset: list):
        """Compute groups for data splitting.
        
        Parameters
        ----------
        dataset : list
            List of Data-objects
        
        Returns
        ----------
        list
            A list of group indices.
        """
        
        keys = default_config.get("grouping_keys", [])
        labels = ["_".join([str(d[k]) for k in keys]) for d in dataset]
        
        unique_labels = list(set([str(label) for label in labels]))
        group_dict = {label:(idx+1) for idx, label in enumerate(unique_labels)}
        
        return [group_dict[label] for label in labels]
    
    @staticmethod
    def stopping_criterion(trial_id, result, threshold=0, key="validation_loss", relation="lt"):
        """A function determining if the training shall be stopped.
        
        Parameters
        ----------
        trial_id : str
            Identifier of a calling trial
        result : dict
            A dictionary of intermediate result scores
        threshold : int
            A threshold that needs to be hit (Default 0)
        key : str
            The key in the result dictionary (Default 'validation_loss')
        relation : str
            If the value needs to be smaller or greater (Default 'lt')
            
        Returns
        ----------
        bool 
            Whether the training shall be stopped
        """
        return result[key] < threshold if relation == "lt" else result[key] > threshold
    
    @staticmethod
    def split_data(dataset: list):
        """Split data into training and validation data.
        
        Parameters
        ----------
        dataset : list
            List of Data-objects
        
        Returns
        ----------
        list, list
            Two lists of Data-objects
        """
        
        groups = np.array(HPTuner.compute_groups(dataset)) 
        logging.info(f"#groups for stratified-split: {len(np.unique(groups))}")
                        
        train_indices, val_indices = train_test_split(np.arange(len(dataset)), # needed, but not used
                                                      test_size=0.2, 
                                                      shuffle=True, 
                                                      stratify=groups) 
        
        train_list : list = [dataset[i] for i in train_indices]
        val_list : list = [dataset[i] for i in val_indices]
        
        return train_list, val_list
    
    @staticmethod
    def perform_tuning(hptuner_instance, epochs : int, dataset: list):
        """Perform hyperparameter optimization.
        
        Parameters
        ----------
        hptuner_instance : HPTuner
            An hp-instance to use for hyperparameter optimization
        epochs: int
            Max. numbers of epoch to train
        dataset: list
            List of Data-objects
        
        Returns
        ----------
        dict, dict, dict, dict, float, float
            The model state, optimizer state, trainer state, optimal training configuration, 
            the smallest achieved validation loss and the time it took for training.
        """
        
        scheduler = ASHAScheduler(max_t=epochs,
                                  **default_config.get("scheduler", {}))
        
        reporter = CLIReporter(**default_config.get("reporter", {}))

        search_alg = OptunaSearch(**default_config.get("optuna_search", {}))
        
        search_alg = ConcurrencyLimiter(search_alg, 
                                        **default_config.get("concurrency_limiter", {}))
        
        tune_run_config = default_config.get("tune_run", {})
        tune_run_config["config"] = {k:(tune.choice(v)) for k,v in tune_run_config.get("config", {}).items()}
        
        train_list, val_list = HPTuner.split_data(dataset)
        
        tune_run_name = f"{hptuner_instance.job_identifier}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        start = time.time()
        
        analysis: ExperimentAnalysis = tune.run(
            tune.with_parameters(partial(hptuner_instance, epochs=epochs), train_list=train_list, val_list=val_list),
            name=tune_run_name,
            scheduler=scheduler,
            progress_reporter=reporter,
            search_alg=search_alg,
            stop=partial(HPTuner.stopping_criterion, **default_config.get("stopping_criterion", {})), 
            **tune_run_config)
        
        time_taken = time.time() - start
        
        # get best trial
        best_trial = analysis.get_best_trial(**default_config.get("tune_best_trial", {}))
        
        # get some information from best trial        
        best_trial_val_loss = best_trial.metric_analysis[tune_run_config["metric"]][tune_run_config["mode"]]
        
        logging.info(f"Time taken: {time_taken:.2f} seconds.")
        logging.info("Best trial config: {}".format(best_trial.config))
        logging.info("Best trial final validation loss: {}".format(best_trial_val_loss))

        # load best checkpoint of best trial
        best_checkpoint_dir = analysis.get_best_checkpoint(best_trial)
        
        model_state_dict, optimizer_state_dict, trainer_state_dict = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        
        return {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "trainer_state_dict": trainer_state_dict,
            "best_trial_config": best_trial.config,
            "best_trial_val_loss": best_trial_val_loss,
            "time_taken": time_taken
        }
        
    def __call__(self, config, checkpoint_dir=None, epochs:int = None, train_list:list = [], val_list:list = []):
        """Called by 'tune.run' during hyperparameter optimization. Check: https://docs.ray.io/en/releases-1.1.0/tune/api_docs/trainable.html
        
        Parameters
        ----------
        config : dict
            A new setup to test
        checkpoint_dir: str
            Path to temporary folder for checkpoints
        epochs: int
            Max. numbers of epoch to train
        train_list: list
            A list of Data-objects for training
        val_list: list
            A list of Data-objects for validation
        """
        
        ### extract and override ### 
        batch_size = self.batch_size
        model_args, optimizer_args, training_loss_args = update_flat_dicts(config, [self.model_args, 
                                                                                    self.optimizer_args, 
                                                                                    self.training_loss_args])
        ############################
        
        model = self.model_class(**model_args).double()
        optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_args)
        loss = self.training_loss_class(**training_loss_args)
        
        ### create trainer ###
        trainer = create_supervised_trainer(model, 
                                            optimizer, 
                                            loss_fn=loss, 
                                            device=self.device, 
                                            non_blocking=True
                                            )
        ######################
        
        ### restore if possible ###
        if checkpoint_dir:
            checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.load_state_dict(checkpoint["trainer_state_dict"])
        ###########################
        
        if self.device != "cpu" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        
        ### create evaluator ###
        metric_output_transform_func = lambda output: {"y_pred": output[0]["y_pred"], "y": output[1]["y"]}
        val_metrics : dict = {
            "loss": Loss(loss),
            "mae": MeanAbsoluteError(output_transform=metric_output_transform_func),
            "mse": MeanSquaredError(output_transform=metric_output_transform_func)
        }
        evaluator = create_supervised_evaluator(model, 
                                                device=self.device,
                                                metrics=val_metrics)
        ########################
        
        ### create data loaders ###
        train_loader = DataLoader(train_list, 
                                  shuffle=True, 
                                  batch_size=batch_size, 
                                  follow_batch=self.follow_batch)
        
        val_loader = DataLoader(val_list, 
                                shuffle=False, 
                                batch_size=batch_size, 
                                follow_batch=self.follow_batch)
        ###########################
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def post_epoch_actions(trainer):
            
            # evaluate model on validation set
            evaluator.run(val_loader)
            val_metrics = evaluator.state.metrics
            
            current_epoch : int = trainer.state.epoch
            
            with tune.checkpoint_dir(current_epoch) as checkpoint_dir: 
                
                model_state_dict = None
                if isinstance(model, nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
                # save model, optimizer and trainer checkpoints
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model_state_dict, optimizer.state_dict(), trainer.state_dict()), path)
            
            # report validation scores to ray-tune
            report_dict : dict = {
                default_config.get("tune_run", {}).get("metric", None): val_metrics["loss"],
                "mae": val_metrics["mae"],
                "mse": val_metrics["mse"],
                "done": current_epoch == epochs
            }
                            
            tune.report(**report_dict)
        
        # start training
        trainer.run(train_loader, max_epochs=epochs)