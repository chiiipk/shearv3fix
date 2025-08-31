#!/bin/bash

# Quick finetuning script for pruned BGE-M3 model
# Usage: ./run_finetune.sh [model_path]

MODEL_PATH=${1:-"../bge_pruning/experiments_1nua/production_hf"}

echo "🚀 Starting BGE-M3 Finetuning"
echo "📁 Model path: $MODEL_PATH"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path $MODEL_PATH not found"
    exit 1
fi

# Using HuggingFace datasets - no local data directory needed

# Run training with optimal settings
python train.py \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --unfreeze_layers 2 \
    --max_length 512 \
    --save_dir checkpoints

echo "✅ Finetuning completed!"
echo "📁 Best model saved in: checkpoints/best_model"
