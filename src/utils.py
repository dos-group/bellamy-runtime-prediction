import logging
from importlib import reload
import os
import yaml
from yaml import safe_load
import copy


def create_dirs(path: str):
    """Creates a directory, recursively if needed.

    Parameters
    ----------
    path : str
        A path that needs to be created
    """
    
    try:
        os.makedirs(path)
    except:
        pass
    

def update_flat_dicts(root: dict, targets: list):
    """Initializes logging with specific settings.

    Parameters
    ----------
    root : dict
        The source config-dict
    targets: list
        A list of config-dicts which need to be updated
        
    Returns
    ----------
    list
        a list of config-dicts
    """
    
    my_root = copy.deepcopy(root)
    my_targets = [copy.deepcopy(target) for target in targets]
    
    for k,v in my_root.items():
        for target in my_targets:
            if k in target:
                target[k] = v
    
    return my_targets


def init_logging(logging_level: str):
    """Initializes logging with specific settings.

    Parameters
    ----------
    logging_level : str
        The desired logging level
    """
    
    reload(logging)
    logging.basicConfig(
        level=getattr(logging, logging_level.upper()),
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s](%(name)s)__[L%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    logging.info(f"Successfully initialized logging.")
    

def load_config():
    """The config to use.

    Returns
    ----------
    dict
        The loaded config
    """
    
    path_to_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    path_to_file = os.path.join(path_to_dir, "config.yml")
    
    def adapt_paths(dictt: dict):
        for key, value in dictt.items():
            if "_dir" in key:
                dictt[key] = os.path.join(path_to_dir, value)
            elif isinstance(value, dict):
                dictt[key] = adapt_paths(value)
            else: 
                pass
            
        return dictt
    
    default_config = {}
    with open(path_to_file, 'r') as stream:
        try:
            default_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
    
    default_config = adapt_paths(default_config)
    return default_config



default_config = load_config()
    
    
