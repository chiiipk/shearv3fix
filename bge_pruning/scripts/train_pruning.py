import torch
import argparse
import yaml
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from composer import Trainer
from composer.utils import reproducibility
from omegaconf import OmegaConf

from models import ComposerBGEM3, get_model_class
from datasets import create_mixed_dataloader, create_sts_dataloader, create_mteb_dataloader
from callbacks import EmbeddingCallback, EvaluationCallback, PruningCallback

def setup_dataloaders(cfg):
    """Setup train and eval dataloaders based on config"""
    dataloaders = {}
    
    # Train dataloader
    train_cfg = cfg.train_loader
    if train_cfg.name == 'mixed':
        train_dataloader = create_mixed_dataloader(
            sts_path=train_cfg.dataset.sts_path,
            mteb_path=train_cfg.dataset.mteb_path,
            batch_size=train_cfg.batch_size,
            sts_ratio=train_cfg.dataset.get('sts_ratio', 0.5),
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
            shuffle=True
        )
    elif train_cfg.name == 'sts':
        train_dataloader = create_sts_dataloader(
            data_path=train_cfg.dataset.sts_path,
            batch_size=train_cfg.batch_size,
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
            shuffle=True
        )
    elif train_cfg.name == 'mteb':
        train_dataloader = create_mteb_dataloader(
            data_path=train_cfg.dataset.mteb_path,
            batch_size=train_cfg.batch_size,
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
            shuffle=True
        )
    else:
        raise ValueError(f"Unknown train dataloader type: {train_cfg.name}")
    
    dataloaders['train'] = train_dataloader
    
    # Eval dataloader
    eval_cfg = cfg.eval_loader
    if eval_cfg.name == 'sts':
        eval_dataloader = create_sts_dataloader(
            data_path=eval_cfg.dataset.sts_path,
            batch_size=eval_cfg.batch_size,
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
            shuffle=False
        )
    elif eval_cfg.name == 'mteb':
        eval_dataloader = create_mteb_dataloader(
            data_path=eval_cfg.dataset.mteb_path,
            batch_size=eval_cfg.batch_size,
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
            shuffle=False
        )
    elif eval_cfg.name == 'mixed':
        eval_dataloader = create_mixed_dataloader(
            sts_path=eval_cfg.dataset.sts_path,
            mteb_path=eval_cfg.dataset.mteb_path,
            batch_size=eval_cfg.batch_size,
            sts_ratio=eval_cfg.dataset.get('sts_ratio', 0.5),
            tokenizer_name=cfg.tokenizer.name,
            max_length=cfg.tokenizer.max_length,
            shuffle=False
        )
    else:
        raise ValueError(f"Unknown eval dataloader type: {eval_cfg.name}")
    
    dataloaders['eval'] = eval_dataloader
    
    return dataloaders

def setup_model(cfg):
    """Setup model based on config"""
    model_class = get_model_class(cfg.model.name)
    model = model_class(cfg.model)
    return model

def setup_callbacks(cfg):
    """Setup callbacks based on config"""
    callbacks = []
    
    if hasattr(cfg, 'callbacks') and cfg.callbacks:
        for callback_cfg in cfg.callbacks:
            if callback_cfg.name == 'embedding_callback':
                callbacks.append(EmbeddingCallback())
            elif callback_cfg.name == 'evaluation_callback':
                callbacks.append(EvaluationCallback())
            elif callback_cfg.name == 'pruning_callback':
                callbacks.append(PruningCallback())
    
    return callbacks

def setup_optimizer(model, cfg):
    """Setup optimizer based on config"""
    optimizer_cfg = cfg.optimizer
    
    # Three-group optimization for L0 pruning
    if hasattr(model, 'l0_module'):
        l0_module = model.l0_module
        
        # Model parameters (excluding L0 masks and Lagrangian multipliers)
        model_params = []
        for name, param in model.named_parameters():
            if not name.startswith('l0_module'):
                model_params.append(param)
        
        # L0 mask parameters
        mask_params = list(l0_module.masks.parameters())
        
        # Lagrangian multipliers
        lambda_params = list(l0_module.lambdas.parameters())
        
        param_groups = [
            {'params': model_params, 'lr': optimizer_cfg.lr, 'weight_decay': optimizer_cfg.weight_decay},
            {'params': mask_params, 'lr': optimizer_cfg.lr * 10, 'weight_decay': 0.0},  # Higher LR for masks
            {'params': lambda_params, 'lr': optimizer_cfg.lr * 100, 'weight_decay': 0.0}  # Highest LR for Lagrangian
        ]
        
        if optimizer_cfg.name == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=optimizer_cfg.betas,
                eps=optimizer_cfg.eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_cfg.name}")
    else:
        # Standard optimization
        if optimizer_cfg.name == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_cfg.lr,
                betas=optimizer_cfg.betas,
                eps=optimizer_cfg.eps,
                weight_decay=optimizer_cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_cfg.name}")
    
    return optimizer

def setup_scheduler(optimizer, cfg):
    """Setup learning rate scheduler"""
    scheduler_cfg = cfg.scheduler
    
    if scheduler_cfg.name == 'linear_decay_with_warmup':
        from composer.optim import LinearWithWarmupScheduler
        scheduler = LinearWithWarmupScheduler(
            t_warmup=scheduler_cfg.t_warmup,
            alpha_f=scheduler_cfg.alpha_f
        )
    else:
        scheduler = None
    
    return scheduler

def main():
    parser = argparse.ArgumentParser(description='Train BGE-M3 with L0 pruning')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    reproducibility.seed_all(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    print(f"Training configuration: {args.config}")
    print(f"Run name: {cfg.run_name}")
    
    # Setup components
    print("Setting up model...")
    model = setup_model(cfg)
    
    print("Setting up dataloaders...")
    dataloaders = setup_dataloaders(cfg)
    
    print("Setting up optimizer...")
    optimizer = setup_optimizer(model, cfg)
    
    print("Setting up scheduler...")
    scheduler = setup_scheduler(optimizer, cfg)
    
    print("Setting up callbacks...")
    callbacks = setup_callbacks(cfg)
    
    # Setup loggers
    loggers = []
    if hasattr(cfg, 'loggers'):
        if hasattr(cfg.loggers, 'wandb'):
            from composer.loggers import WandBLogger
            wandb_logger = WandBLogger(
                project=cfg.loggers.wandb.project,
                name=cfg.loggers.wandb.name
            )
            loggers.append(wandb_logger)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders['train'],
        eval_dataloader=dataloaders['eval'],
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
    print("Starting training...")
    trainer.fit()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
