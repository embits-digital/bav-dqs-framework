from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigManager:
    """
    Central config manager for config files.
    Integrates validation for paths, nested dictionaries and file load.
    """
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}

    @staticmethod
    def require_key(mapping: Dict[str, Any], key: str) -> Any:
        if not isinstance(mapping, dict):
            raise TypeError("mapping must be a dict")
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        if key not in mapping:
            raise KeyError(f"Missing required key: {key}")
        return mapping[key]

    @staticmethod
    def require_path(path: str) -> str:
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return str(p.absolute())

    @staticmethod
    def require_path_key(mapping: Dict[str, Any], dotted_path: str) -> Any:
        if not isinstance(mapping, dict):
            raise TypeError("mapping must be a dict")
        if not isinstance(dotted_path, str) or not dotted_path:
            raise ValueError("dotted_path must be a non-empty string")

        cur: Any = mapping
        parts = dotted_path.split(".")
        for i, part in enumerate(parts):
            if not isinstance(cur, dict):
                raise TypeError(
                    f"Expected dict at path segment {i} while resolving '{dotted_path}', got {type(cur)}"
                )
            if part not in cur:
                raise KeyError(f"Missing required key path: {dotted_path}")
            cur = cur[part]
        return cur

    def get(self, dotted_path: str, is_path: bool = False) -> Any:
        """It searches for a value by validating whether it exists and, optionally, whether it is a valid path."""
        val = self.require_path_key(self._config, dotted_path)
        return self.require_path(val) if is_path else val
    
    def get_float(self, dotted_path: str) -> float:
        """It searches for a value and parses to float."""
        return float(self.require_path_key(self._config, dotted_path))


    @classmethod
    def from_yaml(cls, yaml_path: str) -> ConfigManager:
        """Creates an instance from a YAML file.."""
        valid_path = cls.require_path(yaml_path)
        with open(valid_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(data)

    def update_from_args(self, args: Dict[str, Any]):
        """
        Merges arguments from the CLI (e.g., argparse) into the current configuration.
        Preserves existing keys that were not sent via the CLI.
        """
        def deep_update(source: Dict[str, Any], overrides: Dict[str, Any]):
            for key, value in overrides.items():
                if value is None:
                    continue
                if (
                    key in source 
                    and isinstance(source[key], dict) 
                    and isinstance(value, dict)
                ):
                    deep_update(source[key], value)
                else:
                    source[key] = value
            return source
        
        deep_update(self._config, args)
