"""
File I/O utilities for HistoMoE — handling HDF5, numpy, and JSON files.

Usage
-----
    from histomoe.utils.io import save_numpy, load_numpy
    save_numpy(arr, "embeddings.npy")
    arr = load_numpy("embeddings.npy")
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and all parents) if it does not exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The created (or existing) directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_numpy(array: np.ndarray, path: Union[str, Path]) -> None:
    """Save a NumPy array to a ``.npy`` file.

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    path : str or Path
        Output file path (should end in ``.npy``).
    """
    path = Path(path)
    ensure_dir(path.parent)
    np.save(str(path), array)


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """Load a NumPy array from a ``.npy`` file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npy`` file.

    Returns
    -------
    np.ndarray
        Loaded array.
    """
    return np.load(str(path), allow_pickle=True)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Serialize a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
        Data to serialize.
    path : str or Path
        Output JSON file path.
    indent : int
        JSON indentation level (default: 2).
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint_metadata(
    path: Union[str, Path],
    epoch: int,
    metrics: Dict[str, float],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training checkpoint metadata alongside a model checkpoint.

    Parameters
    ----------
    path : str or Path
        Path to save the metadata JSON.
    epoch : int
        Training epoch number.
    metrics : dict
        Dictionary of evaluation metrics at this checkpoint.
    config : dict, optional
        Experiment configuration snapshot.
    """
    meta = {
        "epoch": epoch,
        "metrics": metrics,
        "config": config or {},
    }
    save_json(meta, path)
