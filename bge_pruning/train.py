import torch
import argparse
from pathlib import Path
import sys
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from composer import Trainer
from composer.utils import reproducibility
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import LinearWithWarmupScheduler
from omegaconf import OmegaConf

# Import BGE components
from models.composer_bge_m3 import ComposerBGEM3
from data_loader import create_dataloader
from callbacks.pruning_callback import PruningCallback

def setup_three_group_optimizer(model, cfg):
    """Setup three-group optimizer for L0 pruning: model params, masks, Lagrangian multipliers"""
    optimizer_cfg = cfg.optimizer
    
    if not hasattr(model, 'l0_module'):
        # Standard single-group optimization
        if optimizer_cfg.name == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_cfg.lr,
                betas=optimizer_cfg.betas,
                eps=optimizer_cfg.eps,
                weight_decay=optimizer_cfg.weight_decay
            )
    
    l0_module = model.l0_module
    
    # Group 1: Model parameters (excluding L0 module)
    model_params = [p for n, p in model.named_parameters() 
                   if not n.startswith('l0_module') and p.requires_grad]
    
    # Group 2: L0 mask parameters
    mask_params = [p for p in l0_module.masks.parameters() if p.requires_grad]
    
    # Group 3: Lagrangian multipliers
    lambda_params = [p for p in l0_module.lambdas.parameters() if p.requires_grad]
    
    # Different learning rates for each group
    param_groups = [
        {
            'params': model_params, 
            'lr': optimizer_cfg.lr, 
            'weight_decay': optimizer_cfg.weight_decay
        },
        {
            'params': mask_params, 
            'lr': optimizer_cfg.lr * 10,  # Higher LR for L0 masks
            'weight_decay': 0.0
        },
        {
            'params': lambda_params, 
            'lr': optimizer_cfg.lr * 100,  # Highest LR for Lagrangian multipliers
            'weight_decay': 0.0
        }
    ]
    
    if optimizer_cfg.name == 'adamw':
        return torch.optim.AdamW(param_groups, betas=optimizer_cfg.betas, eps=optimizer_cfg.eps)
    
    raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")

def main():
    parser = argparse.ArgumentParser(description='BGE-M3 Pruning Training')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry_run', action='store_true', help='Validate setup without training')
    
    args = parser.parse_args()
    
    # Set seed
    reproducibility.seed_all(args.seed)
    
    # Load configuration
    cfg = OmegaConf.load(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Run name: {cfg.run_name}")
    
    # Setup model
    print("Initializing BGE-M3 model with L0 pruning...")
    model = ComposerBGEM3(cfg.model)
    
    # Print model info
    if hasattr(model, 'l0_module'):
        analysis = ModelAnalysis.analyze_model_architecture(model)
        param_counts = ModelAnalysis.count_parameters(model)
        print(f"Model architecture: {analysis.get('base_architecture', 'Unknown')}")
        print(f"Total parameters: {param_counts['total']:,}")
        if 'target_architecture' in analysis:
            print(f"Target architecture: {analysis['target_architecture']}")
    
    # Setup dataloaders
    print("Setting up dataloaders...")
    
    # Training dataloader
    train_cfg = cfg.train_loader
    if train_cfg.name == 'mixed':
        train_dataloader = create_mixed_dataloader(
            sts_path=train_cfg.dataset.sts_path,
            mteb_path=train_cfg.dataset.mteb_path,
            batch_size=train_cfg.batch_size,
            sts_ratio=train_cfg.dataset.get('sts_ratio', 0.5),
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length
        )
    elif train_cfg.name == 'sts':
        train_dataloader = create_sts_dataloader(
            data_path=train_cfg.dataset.sts_path,
            batch_size=train_cfg.batch_size,
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length
        )
    else:
        raise ValueError(f"Unsupported train dataloader: {train_cfg.name}")
    
    # Evaluation dataloader  
    eval_cfg = cfg.eval_loader
    if eval_cfg.name == 'sts':
        eval_dataloader = create_sts_dataloader(
            data_path=eval_cfg.dataset.sts_path,
            batch_size=eval_cfg.batch_size,
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
            shuffle=False
        )
    else:
        eval_dataloader = None
        warnings.warn(f"Unsupported eval dataloader: {eval_cfg.name}")
    
    # Setup optimizer with three groups
    print("Setting up three-group optimizer...")
    optimizer = setup_three_group_optimizer(model, cfg)
    
    # Setup scheduler
    scheduler = None
    if hasattr(cfg, 'scheduler') and cfg.scheduler.name == 'linear_decay_with_warmup':
        from composer.optim import LinearWithWarmupScheduler
        scheduler = LinearWithWarmupScheduler(
            t_warmup=cfg.scheduler.t_warmup,
            alpha_f=cfg.scheduler.alpha_f
        )
    
    # Setup callbacks
    callbacks = []
    if hasattr(cfg, 'callbacks'):
        for callback_cfg in cfg.callbacks:
            if callback_cfg.name == 'embedding_callback':
                callbacks.append(EmbeddingCallback())
            elif callback_cfg.name == 'evaluation_callback':
                callbacks.append(EvaluationCallback())
            elif callback_cfg.name == 'pruning_callback':
                callbacks.append(PruningCallback())
    
    # Setup loggers
    loggers = []
    if hasattr(cfg, 'loggers') and hasattr(cfg.loggers, 'wandb'):
        from composer.loggers import WandBLogger
        wandb_logger = WandBLogger(
            project=cfg.loggers.wandb.project,
            name=cfg.loggers.wandb.name
        )
        loggers.append(wandb_logger)
    
    if args.dry_run:
        print("Dry run completed successfully. Configuration is valid.")
        return
    
    # Create trainer
    print("Creating Composer trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        save_interval=cfg.save_interval,
        save_folder=cfg.save_folder,
        callbacks=callbacks,
        loggers=loggers,
        precision=cfg.get('precision', 'amp_bf16'),
        device_train_microbatch_size=cfg.get('device_train_microbatch_size', 'auto'),
        run_name=cfg.run_name
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting BGE-M3 pruning training...")
    print(f"Max duration: {cfg.max_duration}")
    print(f"Eval interval: {cfg.eval_interval}")
    print(f"Save interval: {cfg.save_interval}")
    
    trainer.fit()
    
    # Final model analysis
    print("\nTraining completed!")
    if hasattr(model, 'l0_module'):
        final_analysis = ModelAnalysis.count_effective_parameters(model)
        print(f"Final effective parameters: {final_analysis['effective']:,}")
        print(f"Parameter reduction: {final_analysis['reduction_ratio']:.2%}")
    
    print(f"Model checkpoints saved to: {cfg.save_folder}")

if __name__ == "__main__":
    main()
