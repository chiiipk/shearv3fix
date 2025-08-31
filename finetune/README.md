# BGE-M3 Pruned Model Finetuning

This package provides a simple and efficient framework for finetuning pruned BGE-M3 models on the STS (Semantic Textual Similarity) dataset.

## Features

- ✅ **Simple**: Clean, minimal codebase focused on STS finetuning
- ✅ **Efficient**: Supports selective layer unfreezing for optimal performance
- ✅ **Flexible**: Configurable training parameters via YAML config
- ✅ **Production-ready**: Direct HuggingFace model loading and saving

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Pruned Model

Ensure your pruned model is saved in HuggingFace format in the `production_hf` directory with these files:
- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `pruning_info.json`

### 3. Data Source

The framework automatically loads the **STS-B (Semantic Textual Similarity Benchmark)** dataset from HuggingFace:
- **Dataset**: `glue/stsb` from HuggingFace datasets
- **Task**: Regression (0-5 similarity scores)
- **Splits**: Train (5,749), Validation (1,500), Test (1,379)
- **No manual data download required** ✅

### 4. Run Finetuning

#### Basic Usage
```bash
python train.py --model_path ../production_hf
```

#### Custom Configuration
```bash
python train.py \
  --model_path ../production_hf \
  --batch_size 32 \
  --epochs 15 \
  --lr 1e-5 \
  --unfreeze_layers 4
```

#### Using Config File
Modify `config.yaml` and run:
```bash
python train.py --model_path ../production_hf --config config.yaml
```

## Configuration Options

### Layer Unfreezing Strategies

- `--unfreeze_layers 0`: Freeze backbone, train only embedding head (fastest)
- `--unfreeze_layers 2`: Unfreeze last 2 layers (balanced)
- `--unfreeze_layers -1`: Full model finetuning (best quality, slowest)

### Key Parameters

- `--batch_size`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 2e-5)
- `--max_length`: Maximum sequence length (default: 512)

## File Structure

```
finetune/
├── __init__.py           # Package initialization
├── model.py              # FinetuneBGEM3 model class
├── data_loader.py        # STS dataset and dataloader
├── train.py              # Main training script
├── config.yaml           # Configuration file
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Training Output

The training script will:

1. **Load the pruned model** from the specified path
2. **Load STS-B dataset** automatically from HuggingFace
3. **Train with progress bars** showing loss and metrics
4. **Save checkpoints** including the best model based on validation correlation
5. **Generate training results** in JSON format

### Output Files

- `checkpoints/best_model/`: Best model based on validation Spearman correlation
- `checkpoints/epoch_N/`: Regular epoch checkpoints
- `checkpoints/training_results.json`: Complete training history and metrics

## Model Performance

The finetuned model optimizes for:
- **Spearman correlation** on STS validation set
- **MSE loss** between predicted and ground truth similarity scores
- **Efficient inference** with the pruned architecture

## Usage After Finetuning

```python
from transformers import AutoModel, AutoTokenizer

# Load finetuned model
model = AutoModel.from_pretrained("checkpoints/best_model")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/best_model")

# Use for sentence similarity
inputs = tokenizer(["sentence 1", "sentence 2"], return_tensors="pt", padding=True)
embeddings = model(**inputs).last_hidden_state[:, 0]  # CLS token
similarity = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
```

## Tips

1. **Start with 2-layer unfreezing** for good balance of speed/quality
2. **Use smaller learning rates** (1e-5) for more layers unfrozen
3. **Monitor validation correlation** as the primary metric
4. **Increase epochs** if validation correlation is still improving

## Technical Details

- **Architecture**: Uses CLS token embeddings with L2 normalization
- **Loss**: MSE loss on cosine similarity scores scaled to [0,5] range
- **Metrics**: Spearman correlation for evaluation
- **Optimization**: AdamW with gradient clipping and weight decay
