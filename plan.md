# BGE-M3 Pruning Project: Complete Implementation Report

## Project Structure Overview

Based on the LLM-Shearing repository structure , I present a complete project structure for pruning BGE-M3 with targeted structured pruning:

```
bge_m3_pruning/
├── bge_pruning/
│   ├── models/
│   │   ├── composer_bge_m3.py           # Main BGE-M3 composer model
│   │   ├── l0_module_embedding.py       # Adapted L0 module for embeddings
│   │   ├── bge_m3_backbone.py          # Masked BGE-M3 implementation
│   │   ├── embedding_heads.py          # Dense/sparse/multi-vector heads
│   │   └── model_registry.py           # Model registration
│   ├── datasets/
│   │   ├── sts_dataloader.py           # STS benchmark data loading
│   │   ├── mteb_dataloader.py          # MTEB retrieval data loading
│   │   ├── contrastive_dataset.py      # Contrastive learning dataset
│   │   └── mixed_dataloader.py         # Combined STS + MTEB loading
│   ├── configs/
│   │   ├── bge_m3/
│   │   │   ├── base.yaml               # Base BGE-M3 configuration
│   │   │   ├── target_75pct.yaml       # 75% parameter target
│   │   │   ├── target_60pct.yaml       # 60% parameter target
│   │   │   └── target_50pct.yaml       # 50% parameter target
│   │   └── training/
│   │       ├── sts_only.yaml           # STS-only training
│   │       ├── mteb_only.yaml          # MTEB-only training
│   │       └── mixed.yaml              # Combined training
│   ├── callbacks/
│   │   ├── embedding_callback.py       # Embedding-specific monitoring
│   │   ├── evaluation_callback.py      # STS + MTEB evaluation
│   │   └── pruning_callback.py         # Adapted pruning callback
│   ├── utils/
│   │   ├── embedding_metrics.py        # STS correlation, MTEB metrics
│   │   ├── model_analysis.py          # Parameter counting, efficiency
│   │   └── visualization.py           # Mask visualization tools
│   ├── scripts/
│   │   ├── train_pruning.py           # Main training script
│   │   ├── evaluate_model.py          # Standalone evaluation
│   │   ├── extract_pruned_model.py    # Export final pruned model
│   │   └── benchmark_efficiency.py    # Performance benchmarking
│   └── train.py                       # Main entry point
├── data/
│   ├── sts/                          # STS benchmark data
│   ├── mteb/                         # MTEB dataset samples
│   └── processed/                    # Preprocessed data cache
├── experiments/
│   ├── logs/                         # Training logs
│   ├── checkpoints/                  # Model checkpoints
│   └── results/                      # Evaluation results
├── tests/
│   ├── test_l0_module.py            # L0 module unit tests
│   ├── test_model_shapes.py         # Architecture validation
│   └── test_dataloaders.py          # Data pipeline tests
├── requirements.txt
├── setup.py
└── README.md
```

## Core Implementation Files

### 1. L0 Module Adaptation (`bge_pruning/models/l0_module_embedding.py`)

The foundation is the L0Module from LLM-Shearing [2](#0-1) , adapted for BGE-M3's XLM-RoBERTa architecture. The key changes include:

- Modified model info setup for XLM-RoBERTa dimensions (1024 hidden, 4096 intermediate, 16 heads, 24 layers)
- Embedding-specific parameter counting that accounts for BGE-M3's output heads
- Target architecture constraints matching the proposed 75%/60%/50% parameter configurations

The hard concrete mask implementation [3](#0-2)  remains largely unchanged, providing the same continuous relaxation for discrete mask optimization.

### 2. BGE-M3 Composer Model (`bge_pruning/models/composer_bge_m3.py`)

Following the pattern established in the LLaMA composer model [4](#0-3) , this implements:

- BGE-M3 backbone with L0 module integration
- Embedding-specific loss functions (STS regression + contrastive InfoNCE)
- Multi-output interface preservation (dense, sparse, multi-vector embeddings)
- Metrics computation for both STS correlation and MTEB retrieval performance

### 3. Training Framework (`bge_pruning/train.py`)

Based on the main training script [5](#0-4) , with key modifications:

- **Optimizer Configuration**: Adapted the three-group parameter setup [6](#0-5)  for embedding model parameters, L0 masks, and Lagrangian multipliers
- **Data Loading**: Custom embedding dataloaders for STS and MTEB datasets instead of text generation
- **Evaluation**: Embedding-specific evaluation metrics (Spearman correlation, nDCG@10, MRR@10)

### 4. Configuration System

Following the YAML-based configuration pattern [7](#0-6) , with embedding-specific configurations:

**Base Configuration (`configs/bge_m3/base.yaml`)**:
```yaml
model:
  name: composer_bge_m3
  base_model: "BAAI/bge-m3"
  d_model: 1024
  n_heads: 16
  n_layers: 24
  intermediate_size: 4096
  vocab_size: 250002
  l0_module:
    pruning_modules: ["layer", "head", "intermediate", "hidden"]
    target_model:
      d_model: 768        # Target hidden size
      n_heads: 12         # Target attention heads  
      n_layers: 18        # Target layers
      intermediate_size: 3072  # Target intermediate size
```

### 5. Embedding Datasets (`bge_pruning/datasets/`)

Custom data loading implementations for:

- **STS Dataset**: Sentence pair regression with similarity scores [0,5]
- **MTEB Dataset**: Query-passage retrieval with contrastive learning
- **Mixed Dataset**: Combined training with configurable mixing ratios

### 6. Callbacks and Monitoring (`bge_pruning/callbacks/`)

Adapted from the pruning callback framework [8](#0-7)  to include:

- Embedding performance monitoring (STS correlation tracking)
- Retrieval performance evaluation (MTEB metrics)
- Mask sparsity visualization
- Parameter efficiency reporting

## Training Protocol

### Phase 1: Initialization
1. Load pre-trained BGE-M3 weights
2. Initialize L0 masks with hard concrete distributions
3. Set up three-group optimizer (model, masks, Lagrangian multipliers)

### Phase 2: Constrained Optimization
Following the Lagrangian optimization approach [9](#0-8) :

- **Training Steps**: 15,000 steps with mixed STS + MTEB objectives
- **Constraint Enforcement**: Exact parameter count constraints for target architectures
- **Evaluation**: Every 500 steps on held-out STS and MTEB validation sets

### Phase 3: Extraction
- Convert continuous masks to discrete binary decisions
- Extract pruned model weights
- Validate preservation of multi-vector interface

## Key Technical Adaptations

### 1. Architecture Mapping
BGE-M3 uses XLM-RoBERTa architecture vs LLaMA/Pythia in the original implementation. The L0 module parameter counting [10](#0-9)  is adapted for:
- Different attention mechanism (multi-head vs grouped-query)
- RoBERTa vs LLaMA layer normalization patterns
- Vocabulary size differences (250K vs 32K tokens)

### 2. Loss Function Integration
Unlike language modeling loss, embedding training requires:
- Bi-encoder similarity computation for STS tasks
- Contrastive learning with in-batch negatives for retrieval
- Temperature-scaled InfoNCE loss with learnable temperature

### 3. Multi-Vector Interface Preservation
BGE-M3's three output modes (dense, sparse, multi-vector) are preserved by:
- Pruning only the shared encoder backbone
- Maintaining original output head architectures
- Ensuring dimensional compatibility across pruning configurations

## Expected Outcomes

### Performance Targets
- **75% Model (203M params)**: ≥95% of original STS performance, ≥92% MTEB performance
- **60% Model (162M params)**: ≥90% of original STS performance, ≥87% MTEB performance  
- **50% Model (135M params)**: ≥85% of original STS performance, ≥80% MTEB performance

### Efficiency Gains
- **Inference Speed**: 1.3-2x faster inference on single GPU
- **Memory Usage**: 25-50% reduction in GPU memory requirements
- **Throughput**: 30-100% increase in queries per second

## Notes

This implementation leverages the robust L0 module framework  from LLM-Shearing, adapting it specifically for embedding model pruning. The constrained optimization approach ensures exact parameter targets while the hard concrete relaxation enables end-to-end differentiable training. The project structure maintains modularity for easy experimentation with different target architectures and training configurations.

The key innovation is extending structured pruning from generative language models to embedding models, requiring careful preservation of the multi-modal interface while adapting the training objectives for similarity and retrieval tasks.
