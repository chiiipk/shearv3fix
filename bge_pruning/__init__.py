"""BGE-M3 Pruning Package

This package implements structured pruning for BGE-M3 embedding models using the L0 regularization
approach adapted from LLM-Shearing. It provides complete training, evaluation, and model extraction
capabilities for creating efficient embedding models.
"""

from .models import ComposerBGEM3, L0ModuleEmbedding
from .datasets import create_sts_dataloader, create_mteb_dataloader, create_mixed_dataloader
from .callbacks import EmbeddingCallback, EvaluationCallback, PruningCallback
from .utils import EmbeddingMetrics, ModelAnalysis, VisualizationTools

__version__ = "1.0.0"
__author__ = "BGE-M3 Pruning Team"

__all__ = [
    'ComposerBGEM3',
    'L0ModuleEmbedding',
    'create_sts_dataloader',
    'create_mteb_dataloader', 
    'create_mixed_dataloader',
    'EmbeddingCallback',
    'EvaluationCallback',
    'PruningCallback',
    'EmbeddingMetrics',
    'ModelAnalysis',
    'VisualizationTools'
]
