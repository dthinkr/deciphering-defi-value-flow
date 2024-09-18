import os
from types import SimpleNamespace
import yaml


def load_config(config_path="config.yaml"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    full_config_path = os.path.join(project_root, config_path)

    with open(full_config_path, "r") as file:
        config_data = yaml.safe_load(file)
        config = SimpleNamespace(
            **{key.upper(): value for key, value in config_data.items()}
        )
    return config
