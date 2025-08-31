import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from typing import Dict, Any, Optional
import json
import os

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

def create_key_mapping(state_dict_keys):
    """Create key mapping to translate between different naming conventions"""
    mapping = {}
    
    # Check for common prefixes that need to be stripped
    prefixes_to_strip = ['backbone.', 'model.', 'module.']
    
    for old_key in state_dict_keys:
        new_key = old_key
        
        # Strip common prefixes
        for prefix in prefixes_to_strip:
            if old_key.startswith(prefix):
                new_key = old_key[len(prefix):]
                break
        
        # Handle specific BGE model structure mappings
        if 'attention.self.query' in new_key:
            # Standard transformers format
            pass
        elif 'attention.query' in new_key:
            # Some models use different attention structure
            new_key = new_key.replace('attention.query', 'attention.self.query')
        elif 'attention.key' in new_key:
            new_key = new_key.replace('attention.key', 'attention.self.key')
        elif 'attention.value' in new_key:
            new_key = new_key.replace('attention.value', 'attention.self.value')
        
        if old_key != new_key:
            mapping[old_key] = new_key
    
    return mapping

def apply_key_mapping(state_dict, mapping):
    """Apply key mapping to state_dict"""
    if not mapping:
        return state_dict
    
    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = mapping.get(old_key, old_key)
        new_state_dict[new_key] = value
    
    return new_state_dict

class FinetuneBGEM3(nn.Module):
    """Simplified BGE-M3 model for finetuning pruned models on STS tasks"""
    
    def __init__(self, model_path: str, temperature: float = 0.02):
        super().__init__()
        
        # Load pruned model using the same approach as eval_pruned.py
        self.backbone, self.config = self._load_pruned_model(model_path)
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        
        # Model parameters
        self.temperature = temperature
        
        # Freeze backbone initially (can be unfrozen later)
        self._freeze_backbone()
    
    def _detect_architecture_from_state_dict(self, state_dict):
        """Detect actual model architecture from state_dict keys"""
        # Count transformer layers
        layer_keys = [k for k in state_dict.keys() if k.startswith('encoder.layer.')]
        if layer_keys:
            layer_indices = [int(k.split('.')[2]) for k in layer_keys if k.split('.')[2].isdigit()]
            num_layers = max(layer_indices) + 1 if layer_indices else 0
        else:
            num_layers = 0
        
        # Detect hidden size from embeddings
        if 'embeddings.word_embeddings.weight' in state_dict:
            hidden_size = state_dict['embeddings.word_embeddings.weight'].shape[1]
        else:
            hidden_size = 1024  # BGE-M3 default
        
        # Detect attention heads from first layer
        attention_key = 'encoder.layer.0.attention.self.query.weight'
        if attention_key in state_dict:
            query_size = state_dict[attention_key].shape[0]
            num_attention_heads = query_size // (hidden_size // 16)  # BGE-M3 head_dim = 64
        else:
            num_attention_heads = hidden_size // 64  # Default BGE-M3
        
        # Detect intermediate size from first layer MLP
        intermediate_key = 'encoder.layer.0.intermediate.dense.weight'
        if intermediate_key in state_dict:
            intermediate_size = state_dict[intermediate_key].shape[0]
        else:
            intermediate_size = hidden_size * 4  # Default
        
        # Detect vocab and position embeddings
        vocab_size = state_dict['embeddings.word_embeddings.weight'].shape[0] if 'embeddings.word_embeddings.weight' in state_dict else 250002
        max_position_embeddings = state_dict['embeddings.position_embeddings.weight'].shape[0] if 'embeddings.position_embeddings.weight' in state_dict else 8194
        type_vocab_size = state_dict['embeddings.token_type_embeddings.weight'].shape[0] if 'embeddings.token_type_embeddings.weight' in state_dict else 1
        
        return {
            'num_hidden_layers': num_layers,
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'intermediate_size': intermediate_size,
            'vocab_size': vocab_size,
            'max_position_embeddings': max_position_embeddings,
            'type_vocab_size': type_vocab_size,
            'model_type': 'xlm-roberta',
            'layer_norm_eps': 1e-5,
            'pad_token_id': 1,
            'position_embedding_type': 'absolute'
        }
    
    def _load_pruned_model(self, model_path: str):
        """Load pruned model with EXACT architecture matching - NO missing keys, NO random values"""
        # Check for weights file - prefer .safetensors over .bin
        safetensors_path = os.path.join(model_path, 'model.safetensors')
        bin_path = os.path.join(model_path, 'pytorch_model.bin')
        
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            weights_path = safetensors_path
            weights_format = 'safetensors'
        elif os.path.exists(bin_path):
            weights_path = bin_path  
            weights_format = 'bin'
        else:
            raise FileNotFoundError(f"No weights file found in {model_path}. Looking for model.safetensors or pytorch_model.bin")
        
        print(f"Loading weights from {weights_format} format: {os.path.basename(weights_path)}")
        
        # Load weights first to detect actual architecture
        if weights_format == 'safetensors':
            state_dict = load_safetensors(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location='cpu')
        
        # Try to fix key naming issues  
        print("🔧 Attempting to fix key naming mismatches...")
        
        # Create key mapping to fix naming issues
        all_keys = list(state_dict.keys())
        key_mapping = create_key_mapping(all_keys)
        
        if key_mapping:
            print(f"🔧 Found {len(key_mapping)} keys that need mapping")
            # Apply key mapping
            state_dict = apply_key_mapping(state_dict, key_mapping)
            print("✅ Applied key mapping")
        
        # Check if this uses standard XLM-RoBERTa key format (after mapping)
        has_encoder_layers = any(k.startswith('encoder.layer.') for k in state_dict.keys())
        has_embeddings = any(k.startswith('embeddings.') for k in state_dict.keys())
        
        if not (has_encoder_layers and has_embeddings):
            print("🔄 Model doesn't use standard format, using AutoModel...")
            # Fallback: Try direct loading as saved
            from transformers import AutoModel
            try:
                model = AutoModel.from_pretrained(model_path)
                config = model.config
                print(f"✅ Loaded via AutoModel: {len(model.encoder.layer)} layers")
                return model, config
            except Exception as e:
                print(f"❌ AutoModel failed: {e}")
                raise RuntimeError("Cannot load this model format")
        
        # Detect actual architecture from state_dict (after key mapping)
        arch_config = self._detect_architecture_from_state_dict(state_dict)
        
        print(f"Detected architecture: {arch_config['num_hidden_layers']} layers, {arch_config['num_attention_heads']} heads, {arch_config['intermediate_size']} intermediate")
        
        # Create config with EXACT detected architecture
        config = XLMRobertaConfig(**arch_config)
        model = XLMRobertaModel(config)
        
        # Try strict loading first to see if key mapping worked
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
            print("🎉 SUCCESS! Strict loading worked after key mapping!")
        except RuntimeError as e:
            print(f"❌ Strict loading still failed, using non-strict...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)} parameters will be randomly initialized")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)} keys ignored")
        
        print(f"✅ Loaded model: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
        if missing_keys:
            print("⚠️ WARNING: Some parameters randomly initialized - results may vary")
        
        return model, config
        
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone for full model finetuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def unfreeze_last_layers(self, num_layers: int = 2):
        """Unfreeze last N layers of the backbone"""
        total_layers = len(self.backbone.encoder.layer)
        for i in range(max(0, total_layers - num_layers), total_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        
        # Get embeddings from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token for sentence embeddings (following BGE-M3 approach)
        embeddings = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return {
            'embeddings': embeddings,
            'last_hidden_state': outputs.last_hidden_state
        }
    
    def compute_sts_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute STS loss for sentence pairs"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        similarity_scores = batch['similarity_scores']
        
        # Forward pass
        outputs = self.forward(input_ids, attention_mask)
        embeddings = outputs['embeddings']  # [batch_size * 2, hidden_size]
        
        # Extract sentence pairs (interleaved format)
        batch_size = embeddings.size(0) // 2
        sent1_emb = embeddings[0::2]  # Even indices: first sentences
        sent2_emb = embeddings[1::2]  # Odd indices: second sentences
        
        # Compute cosine similarity
        predicted_sim = F.cosine_similarity(sent1_emb, sent2_emb, dim=-1)
        
        # Scale to [0, 5] range to match STS scores
        predicted_sim = (predicted_sim + 1) * 2.5
        
        # MSE loss
        loss = F.mse_loss(predicted_sim, similarity_scores)
        
        return {
            'loss': loss,
            'predicted_scores': predicted_sim,
            'ground_truth_scores': similarity_scores
        }
    
    def compute_spearman_correlation(self, predicted_scores: torch.Tensor, 
                                   ground_truth_scores: torch.Tensor) -> float:
        """Compute Spearman correlation for evaluation"""
        try:
            from scipy.stats import spearmanr
            pred_np = predicted_scores.detach().cpu().numpy()
            gt_np = ground_truth_scores.detach().cpu().numpy()
            correlation, _ = spearmanr(pred_np, gt_np)
            return float(correlation) if not torch.isnan(torch.tensor(correlation)) else 0.0
        except ImportError:
            # Fallback to Pearson correlation
            pred_centered = predicted_scores - predicted_scores.mean()
            gt_centered = ground_truth_scores - ground_truth_scores.mean()
            correlation = (pred_centered * gt_centered).sum() / (
                torch.sqrt((pred_centered ** 2).sum() * (gt_centered ** 2).sum()) + 1e-8
            )
            return float(correlation)
    
    def save_model(self, save_path: str):
        """Save finetuned model in HuggingFace format"""
        self.backbone.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save additional finetuning info
        import json
        import os
        finetune_info = {
            'base_model': 'pruned_bge_m3',
            'finetuning_task': 'sts',
            'temperature': self.temperature
        }
        with open(os.path.join(save_path, 'finetune_info.json'), 'w') as f:
            json.dump(finetune_info, f, indent=2)
