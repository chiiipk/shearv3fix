import torch
from composer.core import Callback, State
from composer.loggers import Logger
from typing import Dict, Any
import numpy as np
from scipy.stats import spearmanr

class EmbeddingCallback(Callback):
    """Callback for monitoring embedding-specific metrics"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
    
    def batch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each training batch"""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self._log_embedding_metrics(state, logger)
    
    def eval_end(self, state: State, logger: Logger) -> None:
        """Called at the end of evaluation"""
        self._log_embedding_metrics(state, logger, prefix="eval_")
    
    def _log_embedding_metrics(self, state: State, logger: Logger, prefix: str = "train_") -> None:
        """Log embedding-specific metrics"""
        model = state.model
        
        # Log L0 sparsity information
        if hasattr(model, 'l0_module'):
            sparsity_info = self._get_sparsity_info(model.l0_module)
            for key, value in sparsity_info.items():
                logger.log_metrics({f"{prefix}sparsity_{key}": value})
        
        # Log model parameter counts
        param_info = self._get_parameter_info(model)
        for key, value in param_info.items():
            logger.log_metrics({f"{prefix}params_{key}": value})
    
    def _get_sparsity_info(self, l0_module) -> Dict[str, float]:
        """Get sparsity information from L0 module"""
        sparsity_info = {}
        
        if hasattr(l0_module, 'get_sparsity_loss'):
            _, expected_sparsity, _ = l0_module.get_sparsity_loss()
            
            for mask_name, sparsity in expected_sparsity.items():
                if isinstance(sparsity, torch.Tensor):
                    sparsity_value = sparsity.mean().item() if sparsity.numel() > 1 else sparsity.item()
                else:
                    sparsity_value = sparsity
                sparsity_info[mask_name] = sparsity_value
        
        return sparsity_info
    
    def _get_parameter_info(self, model) -> Dict[str, int]:
        """Get parameter count information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params
        }
    
    def _compute_sts_correlation(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Spearman correlation for STS predictions"""
        try:
            pred_np = predictions.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            correlation, _ = spearmanr(pred_np, target_np)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0