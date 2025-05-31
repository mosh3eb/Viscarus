"""
Viscarus: A versatile vision model that combines general-purpose capabilities with specialized optimizations.
"""

from .models.model import Viscarus
from .models.multimodal import ViscarusMultiModal
from .models.variants import (
    ViscarusB0,
    ViscarusB1,
    ViscarusB2,
    ViscarusB3,
    ViscarusB4,
    ViscarusB5,
    ViscarusB6,
    ViscarusB7
)
from .optimization.edge_optimizer import EdgeOptimizer
from .training.trainer import ViscarusTrainer
from .interpretability.explain import ViscarusExplainer

__version__ = "0.1.0"
__author__ = "Your Name"

# Convenience imports
__all__ = [
    'Viscarus',
    'ViscarusMultiModal',
    'ViscarusB0',
    'ViscarusB1',
    'ViscarusB2',
    'ViscarusB3',
    'ViscarusB4',
    'ViscarusB5',
    'ViscarusB6',
    'ViscarusB7',
    'EdgeOptimizer',
    'ViscarusTrainer',
    'ViscarusExplainer'
]
