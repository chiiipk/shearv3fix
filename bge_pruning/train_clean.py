#!/usr/bin/env python3
"""
Production BGE-M3 Pruning Training Script
Focused on head/layer/intermediate pruning with MTEB support
"""

import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Fix datasets import conflict by importing before local modules
import datasets as hf_datasets

from composer import Trainer, DataSpec
from composer.utils import reproducibility
from omegaconf import OmegaConf

# Import BGE components
from models.composer_bge_m3 import ComposerBGEM3
from data_loader import create_dataloader
from callbacks.pruning_callback import PruningCallback

def get_num_samples_in_batch(batch):
    """Get actual batch size from interleaved tensor pairs"""
    return batch['input_ids'].size(0) // 2

def split_batch(batch, microbatch_size):
    """Custom batch splitting for interleaved sentence pairs"""
    if microbatch_size is None:
        return [batch]
    
    actual_batch_size = batch['input_ids'].size(0) // 2  # True batch size
    if actual_batch_size <= microbatch_size:
        return [batch]
    
    # Split into microbatches
    microbatches = []
    for start_idx in range(0, actual_batch_size, microbatch_size):
        end_idx = min(start_idx + microbatch_size, actual_batch_size)
        
        # For interleaved tensors, we need indices: [start*2, start*2+1, ..., end*2-1]
        interleaved_indices = []
        for i in range(start_idx, end_idx):
            interleaved_indices.extend([i * 2, i * 2 + 1])
        
        microbatch = {
            'input_ids': batch['input_ids'][interleaved_indices],
            'attention_mask': batch['attention_mask'][interleaved_indices],
        }
        
        # Handle similarity_scores (has original batch dimension)
        if 'similarity_scores' in batch:
            microbatch['similarity_scores'] = batch['similarity_scores'][start_idx:end_idx]
        
        microbatches.append(microbatch)
    
    return microbatches

def setup_optimizer(model, cfg):
    """Production optimizer with stable L0 learning rates"""
    if not hasattr(model, 'l0_module'):
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Stable parameter groups for L0 pruning
    model_params = [p for n, p in model.named_parameters() 
                   if not n.startswith('l0_module') and p.requires_grad]
    mask_params = [p for p in model.l0_module.masks.parameters() if p.requires_grad]
    lambda_params = [p for p in model.l0_module.lambdas.parameters() if p.requires_grad]
    
    # LLM-Shearing style aggressive L0 learning rates
    base_lr = cfg.lr  # 5e-5
    l0_lr = 1.0       # Fixed high LR for L0 masks (1000x base)
    lagrangian_lr = 1.0  # Fixed high LR for Lagrangian multipliers
    
    param_groups = [
        {'params': model_params, 'lr': base_lr, 'weight_decay': cfg.weight_decay},
        {'params': mask_params, 'lr': l0_lr, 'weight_decay': 0.0},  # LLM-Shearing style
        {'params': lambda_params, 'lr': lagrangian_lr, 'weight_decay': 0.0}  # LLM-Shearing style
    ]
    
    return torch.optim.AdamW(param_groups, betas=cfg.get('betas', [0.9, 0.999]), eps=cfg.get('eps', 1e-8))

def main():
    parser = argparse.ArgumentParser(description='BGE-M3 Pruning Training')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='sts', 
                       help='Dataset to use (sts, msmarco, or mteb/dataset-name)')
    
    args = parser.parse_args()
    
    # Set seed
    reproducibility.seed_all(args.seed)
    
    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"🔥 BGE-M3 Production Pruning Training")
    print(f"📁 Config: {args.config}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎯 Pruning: head/layer/intermediate only")
    
    # Setup model
    print("🚀 Loading BGE-M3 with L0 pruning...")
    model = ComposerBGEM3(cfg.model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total parameters: {total_params:,}")
    print(f"📈 Trainable parameters: {trainable_params:,}")
    
    # Setup data with MTEB support
    print(f"📊 Loading {args.dataset} dataset from HuggingFace...")
    train_dataloader = create_dataloader(
        dataset_name=args.dataset,
        split="train",
        batch_size=cfg.get('batch_size', 16),
        max_length=cfg.get('max_length', 512)
    )
    
    # Use test split for datasets that don't have validation
    eval_split = "validation" if args.dataset == "sts" else "test"
    eval_dataloader = create_dataloader(
        dataset_name=args.dataset,
        split=eval_split,
        batch_size=cfg.get('batch_size', 16),
        max_length=cfg.get('max_length', 512)
    )
    
    # Wrap dataloaders in DataSpec for Composer batch size detection and custom splitting
    train_dataspec = DataSpec(
        dataloader=train_dataloader, 
        get_num_samples_in_batch=get_num_samples_in_batch,
        split_batch=split_batch
    )
    eval_dataspec = DataSpec(
        dataloader=eval_dataloader, 
        get_num_samples_in_batch=get_num_samples_in_batch,
        split_batch=split_batch
    )
    
    # Setup optimizer
    optimizer = setup_optimizer(model, cfg.optimizer)
    
    # Setup callbacks
    callbacks = [PruningCallback(log_interval=500)]
    
    # Convert PyTorch device names to Composer device names
    device = cfg.get('device', 'gpu')  # Default to 'gpu' for Composer
    
    # If user specified 'cuda' or torch.cuda.is_available(), use 'gpu'
    if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
        device = 'gpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        # Default fallback
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # Setup trainer with DataSpec for proper batch size detection
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataspec,
        eval_dataloader=eval_dataspec,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        save_interval=cfg.get('save_interval', '500ba'),
        save_folder=cfg.get('save_folder', 'experiments/checkpoints'),
        device=device,
        callbacks=callbacks,
        optimizers=optimizer,
        precision=cfg.get('precision', 'amp_bf16'),
    )
    # Start training
    print("🎯 Starting BGE-M3 pruning training...")
    trainer.fit()
    
    # Print final results
    print(f"✅ Production training completed!")
    if hasattr(model, 'l0_module'):
        zs = model.l0_module()
        print(f"🎯 Pruning results:")
        for mask_name, mask_tensor in zs.items():
            sparsity = (mask_tensor == 0).float().mean().item()
            print(f"  {mask_name}: {sparsity:.1%} sparsity")
    
    # Save pruned model in HuggingFace format
    hf_save_path = cfg.get('save_folder', 'experiments/production') + '_hf'
    model.save_pruned_hf_model(hf_save_path)
    print(f"💾 Model ready for deployment")

if __name__ == "__main__":
    main()
