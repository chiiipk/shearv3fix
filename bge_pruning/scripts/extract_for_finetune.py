#!/usr/bin/env python3
"""
Extract pruned model for finetuning compatibility
Converts ComposerBGEM3 checkpoint to standard HuggingFace format
"""

import torch
import argparse
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

def extract_for_finetune(checkpoint_path: str, output_path: str):
    """Extract pruned model to standard HuggingFace format for finetuning"""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    if checkpoint_path.endswith('.pt'):
        # Composer checkpoint format
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        # Assume HuggingFace format already
        raise ValueError("HuggingFace format checkpoints should be used directly")
    
    # Extract backbone weights (remove l0_module and embedding_heads)
    backbone_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            # Remove 'backbone.' prefix for standard HuggingFace format
            new_key = key[9:]  # len('backbone.') = 9
            backbone_state_dict[new_key] = value
    
    print(f"Extracted {len(backbone_state_dict)} backbone parameters")
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_path, "pytorch_model.bin")
    torch.save(backbone_state_dict, model_path)
    
    # Create minimal config (will be updated after pruning is applied)
    config = {
        "architectures": ["XLMRobertaModel"],
        "model_type": "xlm-roberta",
        "hidden_size": 1024,  # Will be updated if hidden pruning applied
        "num_hidden_layers": 24,  # Will be updated after layer pruning
        "num_attention_heads": 16,  # Will be updated after head pruning  
        "intermediate_size": 4096,  # Will be updated after intermediate pruning
        "vocab_size": 250002,
        "max_position_embeddings": 8192,
        "type_vocab_size": 1,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "position_embedding_type": "absolute",
        "use_cache": True,
        "classifier_dropout": None
    }
    
    # Save config
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Copy tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        tokenizer.save_pretrained(output_path)
        print("✅ Tokenizer saved")
    except Exception as e:
        print(f"⚠️  Could not save tokenizer: {e}")
    
    print(f"✅ Model extracted to {output_path}")
    print("🔧 Note: Load this model with transformers.AutoModel.from_pretrained()")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Extract pruned model for finetuning')
    parser.add_argument('checkpoint_path', help='Path to trained model checkpoint (.pt)')
    parser.add_argument('output_path', help='Path to save extracted model')
    
    args = parser.parse_args()
    
    extract_for_finetune(args.checkpoint_path, args.output_path)

if __name__ == "__main__":
    main()
