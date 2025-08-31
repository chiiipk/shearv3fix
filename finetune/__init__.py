"""
Finetuning package for pruned BGE-M3 models
"""

from .model import FinetuneBGEM3
from .data_loader import create_sts_dataloader, load_sts_dataset, get_sts_info

__version__ = "1.0.0"
__all__ = ["FinetuneBGEM3", "create_sts_dataloader", "load_sts_dataset", "get_sts_info"]
