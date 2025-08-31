#!/usr/bin/env python3
"""
Finetuning script for pruned BGE-M3 model on STS dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import time

from model import FinetuneBGEM3
from data_loader import create_sts_dataloader, load_sts_dataset, get_sts_info

class Trainer:
    """Simple trainer for STS finetuning"""
    
    def __init__(
        self,
        model: FinetuneBGEM3,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            loss_dict = self.model.compute_sts_loss(batch)
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self) -> dict:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_predicted = []
        all_ground_truth = []
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Evaluating")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                loss_dict = self.model.compute_sts_loss(batch)
                loss = loss_dict['loss']
                
                # Collect predictions
                all_predicted.extend(loss_dict['predicted_scores'].cpu().numpy())
                all_ground_truth.extend(loss_dict['ground_truth_scores'].cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute correlation
        predicted_tensor = torch.tensor(all_predicted)
        ground_truth_tensor = torch.tensor(all_ground_truth)
        correlation = self.model.compute_spearman_correlation(predicted_tensor, ground_truth_tensor)
        
        return {
            'loss': avg_loss,
            'spearman_correlation': correlation
        }
    
    def train(self, num_epochs: int, save_best: bool = True) -> dict:
        """Main training loop"""
        best_correlation = -1.0
        training_history = []
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate()
            
            epoch_time = time.time() - start_time
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_correlation': val_metrics['spearman_correlation'],
                'epoch_time': epoch_time
            }
            training_history.append(epoch_results)
            
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Correlation: {val_metrics['spearman_correlation']:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Save best model
            if save_best and val_metrics['spearman_correlation'] > best_correlation:
                best_correlation = val_metrics['spearman_correlation']
                self.save_checkpoint('best_model')
                print(f"  → New best model saved (correlation: {best_correlation:.4f})")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')
            
            print()
        
        return {
            'best_correlation': best_correlation,
            'training_history': training_history
        }
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, name)
        self.model.save_model(checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description='Finetune pruned BGE-M3 on STS dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pruned model (production_hf folder)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--unfreeze_layers', type=int, default=2,
                       help='Number of last layers to unfreeze (0 = freeze all, -1 = unfreeze all)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set all random seeds for reproducibility
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    # Load and display STS dataset info
    print("📊 Loading STS-B dataset from HuggingFace...")
    sts_info = get_sts_info()
    print(f"   Dataset: {sts_info['dataset_name']}")
    print(f"   Task: {sts_info['task_type']}")
    print(f"   Train samples: {sts_info['train_size']:,}")
    print(f"   Validation samples: {sts_info['validation_size']:,}")
    print(f"   Test samples: {sts_info['test_size']:,}")
    
    # Create model
    print(f"Loading model from {args.model_path}...")
    model = FinetuneBGEM3(args.model_path)
    
    # Configure layer freezing
    if args.unfreeze_layers == -1:
        model.unfreeze_backbone()
        print("Unfroze entire backbone for full finetuning")
    elif args.unfreeze_layers > 0:
        model.unfreeze_last_layers(args.unfreeze_layers)
        print(f"Unfroze last {args.unfreeze_layers} layers")
    else:
        print("Backbone frozen - only embedding head will be trained")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader = create_sts_dataloader(
        split="train",
        tokenizer_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True
    )
    
    val_loader = create_sts_dataloader(
        split="validation",
        tokenizer_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False
    )
    
    print(f"✅ Train samples: {len(train_loader.dataset):,}")
    print(f"✅ Val samples: {len(val_loader.dataset):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir
    )
    
    # Train
    results = trainer.train(args.epochs)
    
    # Save final results
    results_path = os.path.join(args.save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed!")
    print(f"Best validation correlation: {results['best_correlation']:.4f}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
