import logging

import os
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

from src.pretraining import Pretrainer
from src.utils import default_config, init_logging


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logging-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO",
                        help="Logging level to use.")
    parser.add_argument("-tt", "--training-target", type=str, choices=["c3o", "bell", "all"], required=True,
                        help="Data to use for pre-training.")
    parser.add_argument("-a", "--algorithm", type=str, choices=["grep", "sort", "pagerank", "sgd", "kmeans"], required=True,
                        help="Algorithm to pretrain a model for.")
    args = parser.parse_args()

    if args.logging_level is not None:
        logging_level = args.logging_level
    else:
        logging_level = "INFO"
    
    init_logging(logging_level)
    
    if args.training_target == "bell" and args.algorithm in ["kmeans", "sort"]:
        logging.error("For Bell Dataset, only 'grep', 'pagerank', and 'sgd' are available.")
        sys.exit()
    
    logging.info("Initialize Pretrainer...")
    pretrainer = Pretrainer(args.training_target)
    
    logging.info(f"Start pretraining on {args.algorithm} using {args.training_target}!")
    pretrainer(args.algorithm)
    
    logging.info("Pretrained pipeline and checkpoint saved. I will shutdown now.")
    
