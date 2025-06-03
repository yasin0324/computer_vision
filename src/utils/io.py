"""
Input/Output utility functions.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Object to save
        filepath: Path to save the pickle file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Object loaded from pickle
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f) 