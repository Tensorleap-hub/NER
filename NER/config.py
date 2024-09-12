import os
from typing import Dict, Any
import yaml


def load_od_config() -> Dict[str, Any]:
    # Load the existing YAML config
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    cache_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), config['cache_dir']))
    config['cache_dir'] = cache_dir
    return config


CONFIG = load_od_config()


