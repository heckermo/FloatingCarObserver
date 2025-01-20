import yaml
from dataclasses import dataclass
from typing import Any, Dict
import logging


def get_config(config_path: str):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = Config(config_dict)
    return config

def config_to_dict(config_path: str): 
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to Config instances
                value = Config(value)
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __repr__(self):
        # Provide a readable representation of the Config object
        return f"Config({self.__dict__})"

def create_logger(path) -> logging.Logger:
    """
    Create and configure a logger for the simulation.

    Args:
        path (str): The file path where the log file will be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()  # Console output
    f_handler = logging.FileHandler(path)  # File output
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger