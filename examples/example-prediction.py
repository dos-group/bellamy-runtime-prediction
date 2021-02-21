import logging

import os
import sys
import argparse
import yaml
from yaml import safe_load

sys.path.append(os.path.join(os.path.abspath(''), "..", ".."))

from src.servlets import BellamyModel
from src.utils import default_config, init_logging


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logging-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO",
                        help="Logging level to use.")
    parser.add_argument("-tt", "--training-target", type=str, choices=["c3o", "bell", "all"], required=True,
                        help="Data to use for pre-training.")
    parser.add_argument("-a", "--algorithm", type=str, choices=["grep", "sort", "pagerank", "sgd", "kmeans"], required=True,
                        help="Algorithm to pretrain a model for.")
    parser.add_argument("-r", "--request", type=str, required=True,
                        help="Path to request object.")
    args = parser.parse_args()

    if args.logging_level is not None:
        logging_level = args.logging_level
    else:
        logging_level = "INFO"
    
    init_logging(logging_level)
    
    if args.training_target == "bell" and args.algorithm in ["kmeans", "sort"]:
        logging.error("For Bell Dataset, only 'grep', 'pagerank', and 'sgd' are available.")
        sys.exit()
    
    logging.info("Initialize BellamyModel...")    
    model = BellamyModel(args.training_target)
    
    logging.info("Load request object...")
    payload = {}
    with open(args.request, 'r') as stream:
        try:
            payload = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            sys.exit()
            
    logging.info(f"Received request with data: '{payload}'.")
    
    response_dict = model({**payload, "job_type": args.algorithm})
    
    for k,v in response_dict.items():
        logging.info(f"{k}: {v}")
