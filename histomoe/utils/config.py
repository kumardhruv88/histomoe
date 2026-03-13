"""
Configuration loading and merging utilities using OmegaConf.

Usage
-----
    from histomoe.utils.config import load_config
    cfg = load_config("configs/default.yaml")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(path: Union[str, Path]) -> DictConfig:
    """Load a YAML config file into an OmegaConf DictConfig.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    DictConfig

    Raises
    ------
    FileNotFoundError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(str(path))
    assert isinstance(cfg, DictConfig)
    return cfg


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs (later configs override earlier ones).

    Returns
    -------
    DictConfig
        Merged configuration.
    """
    merged = OmegaConf.merge(*configs)
    assert isinstance(merged, DictConfig)
    return merged


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf DictConfig to a plain Python dict."""
    result = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(result, dict)
    return result   # type: ignore[return-value]


def print_config(cfg: DictConfig, title: Optional[str] = "Configuration") -> None:
    """Pretty-print an OmegaConf config to stdout."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(OmegaConf.to_yaml(cfg))
