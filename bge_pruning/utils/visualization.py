import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Any
import pandas as pd

class VisualizationTools:
    """Visualization tools for mask analysis and model monitoring"""
    
    @staticmethod
    def plot_mask_distribution(mask_values: torch.Tensor, title: str = "Mask Distribution", 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot histogram of mask values"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(mask_values, torch.Tensor):
            values = mask_values.detach().cpu().numpy().flatten()
        else:
            values = np.array(mask_values).flatten()
        
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        ax.set_xlabel('Mask Values')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1 Std: {std_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_sparsity_evolution(sparsity_history: Dict[str, List[float]], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot evolution of sparsity over training"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for mask_name, values in sparsity_history.items():
            ax.plot(values, label=mask_name, linewidth=2)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Sparsity')
        ax.set_title('Sparsity Evolution During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_architecture_comparison(base_arch: Dict[str, int], target_arch: Dict[str, int],
                                   current_arch: Optional[Dict[str, int]] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison of model architectures"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        arch_dims = ['hidden_size', 'num_layers', 'num_attention_heads', 'intermediate_size']
        titles = ['Hidden Size', 'Number of Layers', 'Attention Heads', 'Intermediate Size']
        
        for i, (dim, title) in enumerate(zip(arch_dims, titles)):
            ax = axes[i]
            
            base_val = base_arch.get(dim, 0)
            target_val = target_arch.get(dim, 0)
            
            bars = ['Base', 'Target']
            values = [base_val, target_val]
            colors = ['lightblue', 'orange']
            
            if current_arch:
                current_val = current_arch.get(dim, 0)
                bars.append('Current')
                values.append(current_val)
                colors.append('lightgreen')
            
            ax.bar(bars, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title(title)
            ax.set_ylabel('Count')
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j, v + max(values) * 0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_parameter_reduction(param_counts: Dict[str, int], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot parameter reduction visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        categories = list(param_counts.keys())
        values = list(param_counts.values())
        colors = ['lightblue', 'lightgreen', 'lightcoral'][:len(categories)]
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Parameter Counts')
        ax1.set_ylabel('Number of Parameters')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,}', ha='center', va='bottom')
        
        # Pie chart for relative sizes
        if 'total' in param_counts and 'effective' in param_counts:
            effective = param_counts['effective']
            pruned = param_counts['total'] - effective
            
            sizes = [effective, pruned]
            labels = ['Remaining', 'Pruned']
            colors_pie = ['lightgreen', 'lightcoral']
            
            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Parameter Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_training_metrics(metrics_history: Dict[str, List[float]], 
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot training metrics over time"""
        num_metrics = len(metrics_history)
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            if i < len(axes):
                ax = axes[i]
                ax.plot(values, linewidth=2)
                ax.set_title(metric_name.replace('_', ' ').title())
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_embedding_similarity_heatmap(similarity_matrix: torch.Tensor, 
                                         labels: Optional[List[str]] = None,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot heatmap of embedding similarities"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if isinstance(similarity_matrix, torch.Tensor):
            sim_matrix = similarity_matrix.detach().cpu().numpy()
        else:
            sim_matrix = np.array(similarity_matrix)
        
        sns.heatmap(sim_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'label': 'Cosine Similarity'})
        
        ax.set_title('Embedding Similarity Matrix')
        
        if labels:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_pruning_summary_dashboard(l0_module, metrics_history: Dict[str, List[float]],
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive dashboard for pruning progress"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Mask distributions (top row)
        for i, (mask_name, mask) in enumerate(l0_module.masks.items()):
            if i < 4:  # Only plot first 4 masks
                ax = fig.add_subplot(gs[0, i])
                z_loga = mask.z_loga.detach().cpu().numpy().flatten()
                ax.hist(z_loga, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{mask_name} Mask Distribution')
                ax.set_xlabel('Log-alpha values')
                ax.set_ylabel('Frequency')
        
        # 2. Sparsity evolution (middle left)
        ax_sparsity = fig.add_subplot(gs[1, :2])
        for mask_name, mask in l0_module.masks.items():
            _, sparsity = mask.calculate_expected_score_sparsity()
            sparsity_val = sparsity.mean().item()
            # For demo, just plot current sparsity (would use history in real implementation)
            ax_sparsity.bar(mask_name, sparsity_val, alpha=0.7)
        ax_sparsity.set_title('Current Sparsity by Component')
        ax_sparsity.set_ylabel('Sparsity')
        ax_sparsity.tick_params(axis='x', rotation=45)
        
        # 3. Training metrics (middle right)
        ax_metrics = fig.add_subplot(gs[1, 2:])
        if metrics_history:
            for metric_name, values in list(metrics_history.items())[:3]:  # Plot first 3 metrics
                ax_metrics.plot(values, label=metric_name, linewidth=2)
            ax_metrics.set_title('Training Metrics')
            ax_metrics.set_xlabel('Steps')
            ax_metrics.legend()
            ax_metrics.grid(True, alpha=0.3)
        
        # 4. Architecture comparison (bottom)
        ax_arch = fig.add_subplot(gs[2, :])
        if l0_module.target_model_info:
            base_info = l0_module.base_model_info
            target_info = l0_module.target_model_info
            
            dims = ['hidden_size', 'num_layers', 'num_attention_heads', 'intermediate_size']
            base_vals = [getattr(base_info, dim) for dim in dims]
            target_vals = [getattr(target_info, dim) for dim in dims]
            
            x = np.arange(len(dims))
            width = 0.35
            
            ax_arch.bar(x - width/2, base_vals, width, label='Base', alpha=0.7)
            ax_arch.bar(x + width/2, target_vals, width, label='Target', alpha=0.7)
            
            ax_arch.set_title('Architecture Comparison')
            ax_arch.set_xlabel('Dimensions')
            ax_arch.set_ylabel('Count')
            ax_arch.set_xticks(x)
            ax_arch.set_xticklabels([dim.replace('_', ' ').title() for dim in dims])
            ax_arch.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
