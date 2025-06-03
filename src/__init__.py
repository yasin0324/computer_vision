"""
Tomato Spot Disease Fine-grained Recognition Package

This package implements attention mechanism-based fine-grained recognition
for tomato spot diseases using deep learning techniques.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import config
from . import data
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    "config",
    "data", 
    "models",
    "training",
    "evaluation",
    "utils"
] 