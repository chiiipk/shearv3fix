import torch
import numpy as np
from typing import Dict, Any, Optional
from argparse import Namespace

class ModelAnalysis:
    """Model analysis utilities for parameter counting and efficiency"""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    @staticmethod
    def count_effective_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """Count effective parameters after pruning"""
        if not hasattr(model, 'l0_module'):
            return ModelAnalysis.count_parameters(model)
        
        l0_module = model.l0_module
        effective_params = 0
        
        # Count parameters based on L0 masks
        for mask_name, mask in l0_module.masks.items():
            score, sparsity = mask.calculate_expected_score_sparsity()
            remaining_ratio = 1 - sparsity.mean().item()
            
            if mask_name == 'hidden':
                # Count embedding + output head parameters
                vocab_size = l0_module.base_model_info.vocab_size
                hidden_size = l0_module.base_model_info.hidden_size
                
                embedding_params = vocab_size * hidden_size * remaining_ratio
                output_head_params = 3 * hidden_size * hidden_size * remaining_ratio  # 3 heads
                effective_params += embedding_params + output_head_params
                
            elif mask_name == 'layer':
                # Count per-layer parameters
                hidden_size = l0_module.base_model_info.hidden_size
                intermediate_size = l0_module.base_model_info.intermediate_size
                
                per_layer_params = (
                    4 * hidden_size * hidden_size +  # Attention
                    2 * hidden_size * intermediate_size +  # MLP
                    2 * hidden_size  # Layer norms
                )
                
                num_remaining_layers = int(l0_module.base_model_info.num_layers * remaining_ratio)
                effective_params += num_remaining_layers * per_layer_params
                
            elif mask_name == 'head':
                # Count attention head parameters
                num_layers = l0_module.base_model_info.num_layers
                num_heads = l0_module.base_model_info.num_attention_heads
                head_params = l0_module.base_model_info.params_per_head
                
                remaining_heads = int(num_heads * remaining_ratio)
                effective_params += num_layers * remaining_heads * head_params
                
            elif mask_name == 'intermediate':
                # Count intermediate layer parameters
                num_layers = l0_module.base_model_info.num_layers
                intermediate_size = l0_module.base_model_info.intermediate_size
                params_per_dim = l0_module.base_model_info.params_per_intermediate_dim
                
                remaining_dims = int(intermediate_size * remaining_ratio)
                effective_params += num_layers * remaining_dims * params_per_dim
        
        total_params = ModelAnalysis.count_parameters(model)['total']
        reduction_ratio = 1 - effective_params / total_params
        
        return {
            'total': total_params,
            'effective': int(effective_params),
            'pruned': total_params - int(effective_params),
            'reduction_ratio': reduction_ratio
        }
    
    @staticmethod
    def analyze_model_architecture(model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze model architecture details"""
        analysis = {}
        
        if hasattr(model, 'config'):
            config = model.config
            analysis.update({
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_layers': getattr(config, 'num_hidden_layers', None),
                'num_attention_heads': getattr(config, 'num_attention_heads', None),
                'intermediate_size': getattr(config, 'intermediate_size', None),
                'vocab_size': getattr(config, 'vocab_size', None)
            })
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            # Current vs target architecture
            base_info = l0_module.base_model_info
            target_info = l0_module.target_model_info
            
            analysis['base_architecture'] = {
                'hidden_size': base_info.hidden_size,
                'num_layers': base_info.num_layers,
                'num_attention_heads': base_info.num_attention_heads,
                'intermediate_size': base_info.intermediate_size
            }
            
            if target_info:
                analysis['target_architecture'] = {
                    'hidden_size': target_info.hidden_size,
                    'num_layers': target_info.num_layers,
                    'num_attention_heads': target_info.num_attention_heads,
                    'intermediate_size': target_info.intermediate_size
                }
                
                analysis['reduction_targets'] = {
                    'hidden_reduction': 1 - target_info.hidden_size / base_info.hidden_size,
                    'layer_reduction': 1 - target_info.num_layers / base_info.num_layers,
                    'head_reduction': 1 - target_info.num_attention_heads / base_info.num_attention_heads,
                    'intermediate_reduction': 1 - target_info.intermediate_size / base_info.intermediate_size
                }
        
        return analysis
    
    @staticmethod
    def compute_memory_usage(model: torch.nn.Module, input_shape: tuple = (32, 512)) -> Dict[str, float]:
        """Estimate memory usage in MB"""
        param_count = ModelAnalysis.count_parameters(model)['total']
        
        # Parameter memory (assuming float32)
        param_memory = param_count * 4 / (1024 ** 2)  # MB
        
        # Rough estimate of activation memory
        batch_size, seq_len = input_shape
        if hasattr(model, 'config'):
            hidden_size = getattr(model.config, 'hidden_size', 1024)
            num_layers = getattr(model.config, 'num_hidden_layers', 24)
        else:
            hidden_size = 1024
            num_layers = 24
        
        # Activation memory per layer (rough estimate)
        activation_memory_per_layer = batch_size * seq_len * hidden_size * 4 / (1024 ** 2)
        total_activation_memory = activation_memory_per_layer * num_layers
        
        return {
            'parameter_memory_mb': param_memory,
            'activation_memory_mb': total_activation_memory,
            'total_memory_mb': param_memory + total_activation_memory
        }
    
    @staticmethod
    def compare_models(original_model: torch.nn.Module, pruned_model: torch.nn.Module) -> Dict[str, Any]:
        """Compare original and pruned models"""
        orig_params = ModelAnalysis.count_parameters(original_model)
        pruned_params = ModelAnalysis.count_effective_parameters(pruned_model)
        
        orig_memory = ModelAnalysis.compute_memory_usage(original_model)
        pruned_memory = ModelAnalysis.compute_memory_usage(pruned_model)
        
        comparison = {
            'parameter_reduction': {
                'original': orig_params['total'],
                'pruned': pruned_params['effective'],
                'reduction_ratio': 1 - pruned_params['effective'] / orig_params['total'],
                'reduction_count': orig_params['total'] - pruned_params['effective']
            },
            'memory_reduction': {
                'original_mb': orig_memory['total_memory_mb'],
                'pruned_mb': pruned_memory['total_memory_mb'],
                'reduction_ratio': 1 - pruned_memory['total_memory_mb'] / orig_memory['total_memory_mb'],
                'reduction_mb': orig_memory['total_memory_mb'] - pruned_memory['total_memory_mb']
            }
        }
        
        return comparison
    
    @staticmethod
    def estimate_inference_speedup(reduction_ratio: float, architecture_type: str = 'transformer') -> float:
        """Estimate inference speedup based on parameter reduction"""
        # Rough estimates based on empirical observations
        if architecture_type == 'transformer':
            # Transformer models show roughly linear speedup with parameter reduction
            # but with some overhead, so use a conservative estimate
            speedup = 1 + (reduction_ratio * 0.7)  # 70% of theoretical speedup
        else:
            speedup = 1 + (reduction_ratio * 0.5)  # More conservative for other architectures
        
        return min(speedup, 2.0)  # Cap at 2x speedup
