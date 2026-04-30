import yaml
from typing import Any

def load_config(path: str) -> dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)
