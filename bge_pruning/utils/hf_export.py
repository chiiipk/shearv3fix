"""Minimal HuggingFace export utilities for BGE-M3 pruned models"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModel, AutoConfig, AutoTokenizer


def get_layer_count_from_state_dict(state_dict):
    """Extract number of layers from state dict keys"""
    max_layer = -1
    for key in state_dict.keys():
        if "encoder.layer." in key:
            try:
                layer_num = int(key.split("encoder.layer.")[1].split(".")[0])
                max_layer = max(max_layer, layer_num)
            except (ValueError, IndexError):
                continue
    return max_layer + 1 if max_layer >= 0 else 0


def create_hf_config_from_backbone(backbone):
    """Create HuggingFace config from pruned backbone"""
    config = backbone.config
    
    # Get actual dimensions from pruned model
    actual_layers = len(backbone.encoder.layer)
    actual_vocab_size = backbone.embeddings.word_embeddings.weight.shape[0]
    actual_max_pos = backbone.embeddings.position_embeddings.weight.shape[0]
    actual_type_vocab = backbone.embeddings.token_type_embeddings.weight.shape[0]
    
    # Use first layer to get actual head count and intermediate size
    if actual_layers > 0:
        first_layer = backbone.encoder.layer[0]
        actual_heads = first_layer.attention.num_attention_heads
        actual_intermediate = first_layer.intermediate.dense.out_features
    else:
        actual_heads = config.num_attention_heads
        actual_intermediate = config.intermediate_size
    
    return {
        "architectures": ["XLMRobertaModel"],
        "attention_probs_dropout_prob": getattr(config, "attention_probs_dropout_prob", 0.1),
        "bos_token_id": 0,
        "classifier_dropout": None,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", 0.1),
        "hidden_size": config.hidden_size,
        "initializer_range": getattr(config, "initializer_range", 0.02),
        "intermediate_size": actual_intermediate,
        "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-05),
        "max_position_embeddings": actual_max_pos,
        "model_type": "xlm-roberta",
        "num_attention_heads": actual_heads,
        "num_hidden_layers": actual_layers,
        "output_past": True,
        "pad_token_id": 1,
        "position_embedding_type": "absolute",
        "type_vocab_size": actual_type_vocab,
        "use_cache": True,
        "vocab_size": actual_vocab_size
    }


def convert_backbone_to_hf_state_dict(backbone_state_dict):
    """Convert backbone state dict keys to HuggingFace format"""
    hf_state_dict = {}
    
    for key, value in backbone_state_dict.items():
        # Convert attention layer keys: .attention.query → .attention.self.query
        if ".attention.query." in key:
            new_key = key.replace(".attention.query.", ".attention.self.query.")
        elif ".attention.key." in key:
            new_key = key.replace(".attention.key.", ".attention.self.key.")
        elif ".attention.value." in key:
            new_key = key.replace(".attention.value.", ".attention.self.value.")
        elif ".attention.out_proj." in key:
            new_key = key.replace(".attention.out_proj.", ".attention.output.dense.")
        else:
            new_key = key
            
        hf_state_dict[new_key] = value
    
    return hf_state_dict


def save_backbone_as_hf_model(backbone, save_path, base_model_name="BAAI/bge-m3"):
    """Save pruned backbone as HuggingFace XLM-RoBERTa model"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Create proper HF config
    hf_config_dict = create_hf_config_from_backbone(backbone)
    
    # Create config from base model and update with pruned dimensions
    config = AutoConfig.from_pretrained(base_model_name)
    for key, value in hf_config_dict.items():
        setattr(config, key, value)
    
    # Create fresh HF model with pruned config
    hf_model = AutoModel.from_config(config)
    
    # Convert backbone state dict to HF format
    backbone_state = backbone.state_dict()
    hf_state_dict = convert_backbone_to_hf_state_dict(backbone_state)
    
    # Load weights into HF model
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    
    # Report any remaining key mismatches
    if len(missing_keys) > 0:
        print(f"Warning: {len(missing_keys)} missing keys (may be normal for pruned model)")
    if len(unexpected_keys) > 0:
        print(f"Warning: {len(unexpected_keys)} unexpected keys (may be normal for custom backbone)")
    
    # Save model and config
    hf_model.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(save_path)
    
    return save_path
