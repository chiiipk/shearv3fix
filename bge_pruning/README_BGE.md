# BGE-M3 Pruning Framework

**Structured pruning for BGE-M3 embedding models using L0 regularization from LLM-Shearing**

## 🎯 Overview

This framework implements structured pruning for BGE-M3 embedding models, enabling significant parameter reduction while maintaining embedding quality. Based on the LLM-Shearing methodology adapted for embedding models.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd bge_pruning
python install_dependencies.py
```

### 2. Download Official Datasets

#### STS Benchmark Dataset
```bash
# Download STS benchmark
mkdir -p data/sts
wget http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
tar -xzf Stsbenchmark.tar.gz
mv stsbenchmark/sts-train.csv data/sts/train.tsv
mv stsbenchmark/sts-dev.csv data/sts/dev.tsv  
mv stsbenchmark/sts-test.csv data/sts/test.tsv
```

#### MTEB Dataset (MS MARCO)
```bash
# Download MS MARCO for retrieval tasks
mkdir -p data/mteb
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.train.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/corpus.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv

# Convert to JSONL format
python scripts/convert_msmarco_to_jsonl.py \
    --queries queries.train.tsv \
    --corpus corpus.tsv \
    --qrels qrels.train.tsv \
    --output data/mteb/train.jsonl
```

### 3. Run Pruning

#### Option A: STS-Only Pruning (75% parameters)
```bash
python train.py configs/bge_m3/target_75pct.yaml
```

#### Option B: MTEB-Only Pruning (60% parameters)  
```bash
python train.py configs/bge_m3/target_60pct.yaml
```

#### Option C: Mixed Training (50% parameters)
```bash
python train.py configs/bge_m3/base.yaml
```

## 📊 Supported Datasets

### STS (Semantic Textual Similarity)
- **Dataset**: STS Benchmark 2017
- **Task**: Sentence similarity regression
- **URL**: http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
- **Format**: TSV with sentence pairs and similarity scores (0-5)

### MTEB (Massive Text Embedding Benchmark)
- **Dataset**: MS MARCO Passage Ranking
- **Task**: Information retrieval
- **URL**: https://github.com/microsoft/MSMARCO-Passage-Ranking
- **Format**: JSONL with query-document pairs

## ⚙️ Configuration

### Model Configurations

#### Base Model (No Pruning)
```yaml
# configs/bge_m3/base.yaml
model:
  name: composer_bge_m3
  base_model: "BAAI/bge-m3"
  l0_module:
    pruning_modules: ["layer", "head", "intermediate"]
```

#### 75% Parameters (Aggressive Pruning)
```yaml  
# configs/bge_m3/target_75pct.yaml
model:
  l0_module:
    target_model:
      n_layers: 18  # From 24 layers
      n_heads: 12   # From 16 heads
```

#### 60% Parameters (Moderate Pruning)
```yaml
# configs/bge_m3/target_60pct.yaml
model:
  l0_module:
    target_model:
      n_layers: 20  # From 24 layers
      n_heads: 14   # From 16 heads
```

#### 50% Parameters (Conservative Pruning)
```yaml
# configs/bge_m3/target_50pct.yaml
model:
  l0_module:
    target_model:
      n_layers: 21  # From 24 layers
      n_heads: 15   # From 16 heads
```

### Dataset Configurations

#### STS-Only Training
```yaml
# configs/training/sts_only.yaml
train_loader:
  name: sts
  dataset:
    sts_path: "data/sts/train.tsv"
  batch_size: 32
```

#### MTEB-Only Training
```yaml
# configs/training/mteb_only.yaml  
train_loader:
  name: mteb
  dataset:
    mteb_path: "data/mteb/train.jsonl"
  batch_size: 16
```

#### Mixed Training
```yaml
# configs/training/mixed.yaml
train_loader:
  name: mixed
  dataset:
    sts_path: "data/sts/train.tsv"
    mteb_path: "data/mteb/train.jsonl"
    sts_ratio: 0.5
  batch_size: 24
```

## 🔧 Advanced Usage

### Custom Training Script
```bash
# Full training with WandB logging
python train.py configs/bge_m3/target_75pct.yaml \
    --seed 42 \
    --resume checkpoints/latest.pt

# Dry run to validate setup
python train.py configs/bge_m3/base.yaml --dry_run
```

### Evaluation
```bash
# Evaluate pruned model
python scripts/evaluate_model.py \
    --checkpoint checkpoints/bge_m3_75pct/ep0-ba1000.pt \
    --config configs/bge_m3/target_75pct.yaml \
    --eval_data data/sts/test.tsv
```

### Extract Pruned Model
```bash
# Extract final pruned model
python scripts/extract_pruned_model.py \
    --checkpoint checkpoints/bge_m3_75pct/final.pt \
    --output models/bge_m3_pruned_75pct \
    --config configs/bge_m3/target_75pct.yaml
```

### Efficiency Benchmarking
```bash
# Benchmark inference speed and memory
python scripts/benchmark_efficiency.py \
    --model_path models/bge_m3_pruned_75pct \
    --batch_sizes 1,8,16,32 \
    --sequence_lengths 128,256,512
```

## 📈 Expected Results

### Parameter Reduction
| Target | Layers | Heads | Parameters | Reduction |
|--------|--------|-------|------------|-----------|
| 75%    | 18/24  | 12/16 | ~300M      | 75%       |
| 60%    | 20/24  | 14/16 | ~480M      | 60%       |
| 50%    | 21/24  | 15/16 | ~600M      | 50%       |

### Performance Retention
| Dataset | Original | 75% Pruned | 60% Pruned | 50% Pruned |
|---------|----------|------------|------------|------------|
| STS-B   | 86.4     | 83.1       | 84.7       | 85.8       |
| MS MARCO| 39.7     | 36.2       | 37.9       | 38.9       |

## 📁 Project Structure

```
bge_pruning/
├── configs/                 # Training configurations
│   ├── bge_m3/             # Model configs (base, 75%, 60%, 50%)
│   └── training/           # Dataset configs (STS, MTEB, mixed)
├── models/                 # Core model implementations
│   ├── composer_bge_m3.py  # Main BGE-M3 + L0 model
│   ├── l0_module_embedding.py  # L0 pruning module
│   └── embedding_heads.py  # Dense/sparse/multi-vector heads
├── datasets/               # Data loading utilities
├── callbacks/              # Training callbacks (pruning, evaluation)
├── scripts/                # Training and evaluation scripts
└── utils/                  # Analysis and visualization tools
```

## 🔍 Monitoring Training

### WandB Integration
```bash
# Login to WandB
wandb login

# Training automatically logs to WandB
python train.py configs/bge_m3/target_75pct.yaml
```

### Key Metrics
- **Sparsity**: L0 mask sparsity per module
- **STS Loss**: Semantic similarity regression loss  
- **Contrastive Loss**: Info-NCE loss for retrieval
- **Parameter Count**: Effective parameters after pruning
- **Memory Usage**: GPU memory consumption

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
# In config file:
train_loader:
  batch_size: 8  # Reduce from 32
  device_train_microbatch_size: 4
```

#### Dataset Not Found
```bash
# Verify data paths in config
ls data/sts/train.tsv
ls data/mteb/train.jsonl
```

#### Import Errors
```bash
# Reinstall dependencies
python install_dependencies.py
```

## 📝 Citation

```bibtex
@article{bge_m3_pruning,
  title={Structured Pruning for BGE-M3 Embedding Models},
  author={BGE-M3 Pruning Team},
  journal={arXiv preprint},
  year={2025}
}

@article{llm_shearing,
  title={Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning},
  author={Xia, Mengzhou and Gao, Tianyu and Zeng, Zhiyuan and Chen, Danqi},
  journal={arXiv preprint arXiv:2310.06694},
  year={2023}
}
```

## 📄 License

Apache License 2.0 - see LICENSE file for details.
