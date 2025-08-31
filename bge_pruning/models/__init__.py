"""BGE-M3 Pruning Models Package

This package contains the core models for pruning BGE-M3 embedding models,
adapted from the LLM-Shearing framework with pretrained model integration.
"""

from .composer_bge_m3 import ComposerBGEM3
from .l0_module_embedding import L0ModuleEmbedding, Mask
from .embedding_heads import BGEEmbeddingHeads, DenseEmbeddingHead, SparseEmbeddingHead, MultiVectorEmbeddingHead
from .model_registry import COMPOSER_MODEL_REGISTRY, get_model_class, register_model

__all__ = [
    'ComposerBGEM3',
    'L0ModuleEmbedding', 
    'Mask',
    'BGEEmbeddingHeads',
    'DenseEmbeddingHead',
    'SparseEmbeddingHead',
    'MultiVectorEmbeddingHead',
    'COMPOSER_MODEL_REGISTRY',
    'get_model_class',
    'register_model',
]
