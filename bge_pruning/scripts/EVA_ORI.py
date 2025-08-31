#!/usr/bin/env python3
"""
Evaluation script for original BGE-M3 models
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np
import json
import os
import argparse

def load_original_model(model_name_or_path):
    """Load original BGE-M3 model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    
    print(f"Loaded model: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    print(f"Model type: {type(model).__name__}")
    
    return model, tokenizer

def get_embeddings(model, inputs, pooling_method='cls'):
    """Extract embeddings with different pooling methods"""
    with torch.no_grad():
        outputs = model(**inputs)
        
        if pooling_method == 'cls':
            # Use CLS token (first token)
            embeddings = outputs.last_hidden_state[:, 0]
        elif pooling_method == 'mean':
            # Mean pooling with attention mask
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif pooling_method == 'max':
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            embeddings = torch.max(token_embeddings, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        return F.normalize(embeddings, p=2, dim=-1)

def evaluate_sts(model_name_or_path, batch_size=32, pooling_method='cls'):
    """Evaluate on STS-B validation split"""
    # Load model and tokenizer
    model, tokenizer = load_original_model(model_name_or_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    
    # Load STS-B validation dataset
    dataset = load_dataset("glue", "stsb", split="validation")
    
    predictions, targets = [], []
    
    print(f"Evaluating on {len(dataset)} samples with batch size {batch_size}")
    
    for i in range(0, len(dataset), batch_size):
        if i % (batch_size * 10) == 0:  # Progress indicator
            print(f"Processing batch {i // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")
        
        batch_end = min(i + batch_size, len(dataset))
        batch_items = dataset[i:batch_end]
        
        # Extract data from batch
        sentences1 = batch_items['sentence1']
        sentences2 = batch_items['sentence2']
        labels = batch_items['label']
        
        # Tokenize
        inputs1 = tokenizer(
            sentences1,
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(device)
        
        inputs2 = tokenizer(
            sentences2,
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(device)
        
        # Get embeddings and compute similarity
        emb1 = get_embeddings(model, inputs1, pooling_method)
        emb2 = get_embeddings(model, inputs2, pooling_method)
        sim = F.cosine_similarity(emb1, emb2, dim=-1)
        
        # Scale to [0,5] range (STS-B labels are in 0-5 range)
        predictions.extend(((sim + 1) * 2.5).cpu().numpy())
        targets.extend(labels)
    
    # Compute metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    spearman_corr, _ = spearmanr(predictions, targets)
    pearson_corr = np.corrcoef(predictions, targets)[0, 1]
    mse = np.mean((predictions - targets) ** 2)
    
    return {
        'spearman': float(spearman_corr),
        'pearson': float(pearson_corr), 
        'mse': float(mse),
        'samples': len(predictions),
        'pooling_method': pooling_method
    }

def evaluate_multiple_pooling(model_name_or_path, batch_size=32):
    """Evaluate with different pooling methods"""
    pooling_methods = ['cls', 'mean', 'max']
    all_results = {}
    
    for pooling in pooling_methods:
        print(f"\n=== Evaluating with {pooling.upper()} pooling ===")
        results = evaluate_sts(model_name_or_path, batch_size, pooling)
        all_results[pooling] = results
        print(f"{pooling.upper()} - Spearman: {results['spearman']:.4f}, "
              f"Pearson: {results['pearson']:.4f}, MSE: {results['mse']:.4f}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate original BGE-M3 model on STS-B')
    parser.add_argument('--model_path', default='BAAI/bge-m3', 
                       help='Model name or path (default: BAAI/bge-m3)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--pooling', choices=['cls', 'mean', 'max', 'all'], 
                       default='cls', help='Pooling method or "all" for comparison')
    parser.add_argument('--output_dir', help='Directory to save results')
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model_path}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if args.pooling == 'all':
        results = evaluate_multiple_pooling(args.model_path, args.batch_size)
        
        # Find best pooling method
        best_method = max(results.keys(), key=lambda k: results[k]['spearman'])
        print(f"\nBest pooling method: {best_method.upper()} "
              f"(Spearman: {results[best_method]['spearman']:.4f})")
    else:
        results = evaluate_sts(args.model_path, args.batch_size, args.pooling)
        print(f"\nResults with {args.pooling.upper()} pooling:")
        print(f"Spearman: {results['spearman']:.4f}")
        print(f"Pearson: {results['pearson']:.4f}")
        print(f"MSE: {results['mse']:.4f}")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'eval_results.json')
    else:
        output_path = 'eval_results.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()