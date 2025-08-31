import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models.base import ComposerModel
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from typing import Dict, Optional, Any, Tuple
from transformers import AutoModel, AutoConfig, AutoTokenizer

from .l0_module_embedding import L0ModuleEmbedding
from .embedding_heads import BGEEmbeddingHeads
from .bge_m3_backbone import MaskedBGEM3Backbone

class ComposerBGEM3(ComposerModel):
    """BGE-M3 model with L0 pruning and Composer interface"""
    
    def __init__(self, cfg):
        super().__init__()
        
        # Load pretrained BGE-M3 model and config
        model_name = getattr(cfg, 'base_model', 'BAAI/bge-m3')
        self.base_model_name = model_name  # Store for HF export
        base_model = AutoModel.from_pretrained(model_name)
        self.config = base_model.config
        
        # Create masked backbone with original config
        self.backbone = MaskedBGEM3Backbone(self.config)
        self.backbone.load_state_dict(base_model.state_dict(), strict=False)
        
        # Override config with custom settings if provided
        if hasattr(cfg, 'd_model'):
            self.config.hidden_size = cfg.d_model
        if hasattr(cfg, 'n_layers'):
            self.config.num_hidden_layers = cfg.n_layers
        if hasattr(cfg, 'n_heads'):
            self.config.num_attention_heads = cfg.n_heads
            
        # Initialize embedding heads
        self.embedding_heads = BGEEmbeddingHeads(self.config)
        
        # Fix device string conversion
        device_name = get_device(None).name
        if device_name == "gpu":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate configuration consistency
        self._validate_config()
        
        # Initialize L0 module for pruning with model info
        self.l0_module = L0ModuleEmbedding(cfg, device_name, self.backbone)

        # Loss configurations
        self.use_sts_loss = getattr(cfg, 'use_sts_loss', True)
        self.use_contrastive_loss = getattr(cfg, 'use_contrastive_loss', True)
        self.temperature = getattr(cfg, 'temperature', 0.02)
        
        # Metrics storage
        self.train_metrics = {}
        self.eval_metrics = {}
        self.ref_model = None
        
    def forward(self, batch):
        """Forward pass through the model"""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Get actual batch size from tensor
        actual_batch_size = input_ids.size(0)
        
        # Get L0 masks
        l0_output = self.l0_module()
        
        # Forward through masked backbone with L0 masks
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_z=l0_output.get('layer_z'),
            head_z=l0_output.get('head_z'),
            intermediate_z=l0_output.get('intermediate_z'),
        )
        
        # Get embeddings from heads (dense only for training efficiency)
        embedding_outputs = self.embedding_heads(
            hidden_states=backbone_outputs["last_hidden_state"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sparse=False,
            return_multi_vector=False,
        )
        
        return {
            'embeddings': embedding_outputs,
            'backbone_outputs': backbone_outputs,
            'l0_output': l0_output,
            'batch_size': actual_batch_size,
        }
    
    def eval_forward(self, batch, outputs=None):
        """Evaluation forward pass"""
        return outputs if outputs is not None else self.forward(batch)
    
    def loss(self, outputs, batch):
        """Production loss computation with proper tensor handling"""
        embeddings = outputs['embeddings']
        l0_output = outputs['l0_output']
        # Calculate batch size from interleaved tensor dimensions
        batch_size = batch['input_ids'].size(0) // 2
        
        total_loss = 0.0
        
        # Task-specific loss computation (infer task from batch contents)
        if 'similarity_scores' in batch:
            # STS task
            sts_loss = self.compute_sts_loss(embeddings, batch, batch_size)
            total_loss += sts_loss
        else:
            # Retrieval task
            contrastive_loss = self.compute_contrastive_loss(embeddings, batch_size)
            total_loss += contrastive_loss
        
        # L0 sparsity loss for pruning (scaled to balance with task loss)
        if hasattr(self.l0_module, 'get_sparsity_loss'):
            sparsity_loss, expected_sparsity, expected_score = self.l0_module.get_sparsity_loss()
            constraint_loss = self.compute_constraint_loss(expected_sparsity)
            # Scale pruning losses to match task loss magnitude (typically 1-5)
            pruning_weight = 20.0  # Amplify small sparsity losses to be significant
            total_loss += pruning_weight * (sparsity_loss + constraint_loss)
        
        return total_loss
    
    def compute_sts_loss(self, embeddings: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_size: int) -> torch.Tensor:
        """Production STS loss with proper paired sentence handling"""
        dense_emb = embeddings['dense_embedding']  # [batch_size * 2, embedding_dim]
        similarity_scores = batch['similarity_scores']  # [batch_size]
        
        # Extract sentence pairs from interleaved format
        sent1_emb = dense_emb[0::2]  # Even indices: first sentences [batch_size, embedding_dim]
        sent2_emb = dense_emb[1::2]  # Odd indices: second sentences [batch_size, embedding_dim]
        
        # Compute cosine similarity
        predicted_sim = F.cosine_similarity(sent1_emb, sent2_emb, dim=-1)
        
        # Scale to [0, 5] range to match STS scores
        predicted_sim = (predicted_sim + 1) * 2.5
        
        # MSE loss
        sts_loss = F.mse_loss(predicted_sim, similarity_scores)
        return sts_loss
    
    def compute_contrastive_loss(self, embeddings: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Production contrastive loss for query-passage pairs"""
        dense_emb = embeddings['dense_embedding']  # [batch_size * 2, embedding_dim]
        
        # Extract queries and passages from interleaved format
        query_emb = dense_emb[0::2]  # Even indices: queries [batch_size, embedding_dim]
        passage_emb = dense_emb[1::2]  # Odd indices: passages [batch_size, embedding_dim]
        
        # Compute similarity matrix: queries vs all passages
        similarity_matrix = torch.matmul(query_emb, passage_emb.t()) / self.temperature
        
        # Labels: each query should match its corresponding passage (diagonal)
        labels = torch.arange(batch_size, device=query_emb.device)
        
        # InfoNCE loss
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        return contrastive_loss
    
    def compute_constraint_loss(self, expected_sparsity: Dict[str, torch.Tensor]) -> torch.Tensor:
        """LLM-Shearing style constraint loss with warmup and quadratic penalty"""
        constraint_loss = 0.0
        
        for mask_name, sparsity in expected_sparsity.items():
            if mask_name in self.l0_module.masks:
                mask = self.l0_module.masks[mask_name]
                
                if hasattr(mask, 'target_sparsity') and mask.target_sparsity is not None:
                    # Get warmup target sparsity (gradually increases from 0 to final target)
                    current_target = self.l0_module.get_warmup_target_sparsity(mask.target_sparsity)
                    
                    # LLM-Shearing style: Linear + Quadratic penalty
                    sparsity_diff = sparsity.mean() - current_target
                    linear_penalty = torch.abs(sparsity_diff)
                    quadratic_penalty = sparsity_diff ** 2
                    
                    # Combine linear + quadratic (quadratic is stronger for large violations)
                    constraint_loss += linear_penalty + 5.0 * quadratic_penalty
        
        return constraint_loss
    
    def get_metrics(self, is_train: bool = False) -> Dict[str, Any]:
        """Get metrics for logging"""
        if is_train:
            return self.train_metrics
        else:
            return self.eval_metrics
    
    def prune_params(self, zs: Optional[Dict[str, torch.Tensor]] = None):
        """Prune model parameters based on masks"""
        if zs is None:
            zs = self.l0_module()
        
        # Prune backbone
        self.backbone.prune_params(zs)
        
        # Note: Hidden dimension pruning removed for production version
    
    def get_model_info(self):
        """Get model architecture information"""
        return {
            'base_model_info': self.l0_module.base_model_info,
            'target_model_info': self.l0_module.target_model_info,
            'pruning_modules': self.l0_module.pruning_modules,
        }
    
    def compute_spearman_correlation(self, predicted_scores: torch.Tensor, 
                                   ground_truth_scores: torch.Tensor) -> float:
        """Compute Spearman correlation for STS evaluation"""
        try:
            from scipy.stats import spearmanr
            pred_np = predicted_scores.detach().cpu().numpy()
            gt_np = ground_truth_scores.detach().cpu().numpy()
            correlation, _ = spearmanr(pred_np, gt_np)
            return float(correlation)
        except ImportError:
            # Fallback to Pearson correlation if scipy not available
            pred_centered = predicted_scores - predicted_scores.mean()
            gt_centered = ground_truth_scores - ground_truth_scores.mean()
            correlation = (pred_centered * gt_centered).sum() / (
                torch.sqrt((pred_centered ** 2).sum() * (gt_centered ** 2).sum()) + 1e-8
            )
            return float(correlation)
    
    def extract_pruned_model(self) -> 'ComposerBGEM3':
        """Extract a pruned model with parameters permanently removed"""
        # Get current masks
        zs = self.l0_module()
        
        # Create new config based on pruned dimensions
        pruned_config = self._create_pruned_config(zs)
        
        # Create new model with pruned config
        pruned_model = ComposerBGEM3(pruned_config)
        
        # Copy and prune weights
        self._copy_pruned_weights(pruned_model, zs)
        
        return pruned_model
    
    def _create_pruned_config(self, zs: Dict[str, torch.Tensor]) -> DictConfig:
        """Create configuration for pruned model"""
        # This would create a new config with reduced dimensions
        # Implementation depends on specific pruning strategy
        pass
    
    def _copy_pruned_weights(self, target_model: 'ComposerBGEM3', 
                           zs: Dict[str, torch.Tensor]):
        """Copy weights from current model to pruned model"""
        # This would copy only the non-pruned weights
        # Implementation depends on specific pruning strategy
        pass
    
    def save_pruned_hf_model(self, save_path: str, tokenizer_name: str = None):
        """Save pruned model in HuggingFace format for production use"""
        import sys
        import os
        from pathlib import Path
        
        # Add project root to path for imports
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from utils.hf_export import save_backbone_as_hf_model
        import json
        
        # Use eval mode for deterministic masks
        was_training = self.training
        self.eval()
        
        # Get deterministic masks and apply pruning
        zs = self.l0_module()
        print("\n🎯 Applying pruning masks...")
        for mask_name, mask_tensor in zs.items():
            sparsity = (mask_tensor == 0).float().mean().item()
            print(f"  {mask_name}: {sparsity:.1%} sparsity")
        
        # Actually remove pruned parameters
        self.prune_params(zs)
        self._validate_pruned_model()
        
        # Save backbone as proper HuggingFace model
        print(f"\n💾 Saving pruned model to {save_path}")
        base_model_name = tokenizer_name or getattr(self, 'base_model_name', 'BAAI/bge-m3')
        save_backbone_as_hf_model(self.backbone, save_path, base_model_name)
        
        # Save pruning info
        pruning_info = {
            'pruning_results': {name: float((mask == 0).float().mean()) for name, mask in zs.items()},
            'base_model': base_model_name,
            'final_config': {
                'num_hidden_layers': len(self.backbone.encoder.layer),
                'num_attention_heads': self.backbone.encoder.layer[0].attention.num_attention_heads if len(self.backbone.encoder.layer) > 0 else 0,
                'intermediate_size': self.backbone.encoder.layer[0].intermediate.dense.out_features if len(self.backbone.encoder.layer) > 0 else 0,
                'hidden_size': self.config.hidden_size
            }
        }
        
        with open(os.path.join(save_path, 'pruning_info.json'), 'w') as f:
            json.dump(pruning_info, f, indent=2)
        
        print(f"✅ Pruned model saved in HuggingFace format!")
        print(f"📁 Location: {save_path}")
        print(f"🔧 Usage: model = AutoModel.from_pretrained('{save_path}')")
        
        # Restore training mode
        if was_training:
            self.train()
        
        return save_path
    
    def _validate_config(self):
        """Validate model configuration for mathematical consistency"""
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.config.hidden_size}) must be divisible by "
                f"number of attention heads ({self.config.num_attention_heads}). "
                f"Adjust configuration to use valid combinations."
            )
    
    def _validate_pruned_model(self):
        """Validate pruned model is in correct state for production use"""
        # Validate backbone configuration
        backbone_config = self.backbone.config
        
        # Check layer count consistency
        actual_layers = len(self.backbone.encoder.layer)
        config_layers = backbone_config.num_hidden_layers
        if actual_layers != config_layers:
            raise ValueError(f"Layer count mismatch: actual={actual_layers}, config={config_layers}")
        
        # Check attention head consistency
        if backbone_config.hidden_size % backbone_config.num_attention_heads != 0:
            raise ValueError(f"Invalid attention configuration after pruning: "
                           f"hidden_size={backbone_config.hidden_size}, "
                           f"num_attention_heads={backbone_config.num_attention_heads}")
        
        print(f"✅ Model validation passed: {actual_layers} layers, "
              f"{backbone_config.num_attention_heads} heads, "
              f"{backbone_config.intermediate_size} intermediate size")
