import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f) or {}


def load_configs(
    model_config_path: str = "configs/model.yaml",
    train_config_path: str = "configs/train.yaml"
) -> Dict[str, Any]:
    """
    Load and merge model and training configurations.
    Training config takes precedence over model config for overlapping keys.
    
    Args:
        model_config_path: Path to model.yaml
        train_config_path: Path to train.yaml
    
    Returns:
        Merged configuration dictionary
    """
    # Load individual configs
    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)
    
    # Merge: train config overrides model config
    config = {**model_config, **train_config}
    
    return config


def print_config(config: Dict[str, Any]) -> None:
    """Pretty print configuration."""
    print("\n" + "="*50)
    print("Configuration Loaded:")
    print("="*50)
    for key, value in sorted(config.items()):
        print(f"  {key:<25} : {value}")
    print("="*50 + "\n")


if __name__ == "__main__":
    config = load_configs()
    print_config(config)
