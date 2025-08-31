import torch
from composer.core import Callback, State
from composer.loggers import Logger
from typing import Dict, Any

class PruningCallback(Callback):
    """Minimal callback for pruning monitoring"""
    
    def __init__(self, log_interval: int = 500):
        self.log_interval = log_interval
        self.step_count = 0
    
    def batch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each training batch"""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self._log_pruning_metrics(state, logger)
            self._update_lagrangian_multipliers(state)
    
    def _log_pruning_metrics(self, state: State, logger: Logger) -> None:
        """Log pruning-specific metrics"""
        model = state.model
        
        if not hasattr(model, 'l0_module'):
            return
            
        l0_module = model.l0_module
        
        # Update current step for sparsity warmup
        current_step = state.timestamp.batch.value if hasattr(state.timestamp, 'batch') else 0
        l0_module.update_training_step(current_step)
        
        # Constrain parameters to valid ranges
        l0_module.constrain_parameters()
        
        metrics = {}
        
        # Log actual sparsity using deterministic masks (same as final results)
        zs = l0_module()
        for mask_name, mask_tensor in zs.items():
            # Remove '_z' suffix and calculate actual sparsity
            base_name = mask_name.replace('_z', '')
            sparsity = (mask_tensor == 0).float().mean().item()
            metrics[f'sparsity/{base_name}'] = sparsity
        
        # Log to console
        sparsity_str = ", ".join([f"{k.split('/')[-1]}: {v:.3f}" for k, v in metrics.items()])
        print(f"Step {self.step_count}: Sparsity - {sparsity_str}")
        
        # Log to logger if available
        if logger and hasattr(logger, 'log_metrics'):
            logger.log_metrics(metrics)
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            # Log constraint violations
            constraint_loss = self._compute_constraint_violations(l0_module)
            logger.log_metrics({"train_constraint_loss": constraint_loss})
            
            # Log effective model size
            effective_params = self._compute_effective_params(l0_module)
            logger.log_metrics({"train_effective_params": effective_params})
            
            # Log mask statistics
            mask_stats = self._compute_mask_statistics(l0_module)
            for key, value in mask_stats.items():
                logger.log_metrics({f"train_mask_{key}": value})
    
    def _update_lagrangian_multipliers(self, state: State) -> None:
        """Update Lagrangian multipliers based on constraint violations"""
        model = state.model
        
        if hasattr(model, 'l0_module') and hasattr(model.l0_module, 'lambdas'):
            l0_module = model.l0_module
            current_step = state.timestamp.batch.value
            
            # Only start updating after warmup
            if current_step > l0_module.lagrangian_warmup_steps:
                self._adjust_lagrangian_multipliers(l0_module)
    
    def _adjust_lagrangian_multipliers(self, l0_module) -> None:
        """Adjust Lagrangian multipliers based on constraint satisfaction"""
        learning_rate = 0.001  # Reduced learning rate for stability
        max_lambda = 10.0  # Maximum multiplier value to prevent explosion
        
        for param_name, param in l0_module.lambdas.items():
            if param_name.startswith("lambda_1_"):
                # Adjust based on sparsity constraint
                mask_name = param_name.replace("lambda_1_", "")
                if mask_name in l0_module.masks:
                    mask = l0_module.masks[mask_name]
                    _, sparsity = mask.calculate_expected_score_sparsity()
                    
                    if mask.target_sparsity is not None:
                        violation = sparsity.mean() - mask.target_sparsity
                        param.data += learning_rate * violation
                        param.data.clamp_(min=0.0, max=max_lambda)
    
    def _compute_constraint_violations(self, l0_module) -> float:
        """Compute total constraint violations"""
        total_violation = 0.0
        
        for mask_name, mask in l0_module.masks.items():
            if mask.target_sparsity is not None:
                _, sparsity = mask.calculate_expected_score_sparsity()
                violation = torch.abs(sparsity.mean() - mask.target_sparsity)
                total_violation += violation.item()
        
        return total_violation
    
    def _compute_effective_params(self, l0_module) -> int:
        """Compute effective number of parameters after pruning"""
        total_effective = 0
        
        for mask_name, mask in l0_module.masks.items():
            score, _ = mask.calculate_expected_score_sparsity()
            effective_size = score.sum().item()
            total_effective += int(effective_size * mask.num_params_per_mask)
        
        return total_effective
    
    def _compute_mask_statistics(self, l0_module) -> Dict[str, float]:
        """Compute statistics about mask distributions"""
        stats = {}
        
        for mask_name, mask in l0_module.masks.items():
            z_loga = mask.z_loga
            
            # Mean and std of log-alpha values
            stats[f"{mask_name}_mean_log_alpha"] = z_loga.mean().item()
            stats[f"{mask_name}_std_log_alpha"] = z_loga.std().item()
            
            # Expected sparsity
            _, sparsity = mask.calculate_expected_score_sparsity()
            stats[f"{mask_name}_expected_sparsity"] = sparsity.mean().item()
        
        return stats
    
    def _log_mask_visualizations(self, state: State, logger: Logger) -> None:
        """Create and log mask visualizations"""
        model = state.model
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            for mask_name, mask in l0_module.masks.items():
                # Create histogram of mask values
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    z_loga = mask.z_loga.detach().cpu().numpy()
                    if z_loga.ndim > 1:
                        z_loga = z_loga.flatten()
                    
                    ax.hist(z_loga, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Log-alpha values')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {mask_name} mask values')
                    ax.grid(True, alpha=0.3)
                    
                    # Log the figure
                    logger.log_images({f"mask_distribution_{mask_name}": fig})
                    plt.close(fig)
                except ImportError:
                    pass  # Skip visualization if matplotlib not available
    
    def epoch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each epoch"""
        self._log_pruning_summary(state, logger)
    
    def _log_pruning_summary(self, state: State, logger: Logger) -> None:
        """Log summary of pruning progress"""
        model = state.model
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            # Log target vs current architecture
            if hasattr(l0_module, 'target_model_info') and l0_module.target_model_info is not None:
                target_info = l0_module.target_model_info
                base_info = l0_module.base_model_info
                
                reduction_stats = {
                    'target_hidden_reduction': 1 - target_info.hidden_size / base_info.hidden_size,
                    'target_layer_reduction': 1 - target_info.num_layers / base_info.num_layers,
                    'target_head_reduction': 1 - target_info.num_attention_heads / base_info.num_attention_heads,
                    'target_intermediate_reduction': 1 - target_info.intermediate_size / base_info.intermediate_size
                }
                
                for key, value in reduction_stats.items():
                    logger.log_metrics({f"pruning_{key}": value})