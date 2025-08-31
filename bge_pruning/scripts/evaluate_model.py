import torch
import argparse
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models import ComposerBGEM3
from datasets import create_sts_dataloader, create_mteb_dataloader
from utils.embedding_metrics import EmbeddingMetrics

def evaluate_sts(model, dataloader, device):
    """Evaluate model on STS benchmark"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            similarity_scores = batch['similarity_scores'].to(device)
            
            # Forward pass
            outputs = model({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            
            # Get dense embeddings
            embeddings = outputs['embeddings']['dense_embedding']
            
            # Compute similarity between sentence pairs
            batch_size = embeddings.shape[0] // 2
            sent1_emb = embeddings[:batch_size]
            sent2_emb = embeddings[batch_size:]
            
            predicted_sim = torch.cosine_similarity(sent1_emb, sent2_emb, dim=-1)
            predicted_sim = (predicted_sim + 1) * 2.5  # Scale to [0, 5]
            
            all_predictions.append(predicted_sim)
            all_targets.append(similarity_scores)
    
    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = EmbeddingMetrics.compute_sts_metrics(predictions, targets)
    return metrics

def evaluate_mteb(model, dataloader, device):
    """Evaluate model on MTEB retrieval tasks"""
    model.eval()
    all_query_embeddings = []
    all_doc_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            
            embeddings = outputs['embeddings']['dense_embedding']
            
            # Split triplets (query, positive, negative)
            batch_size = embeddings.shape[0] // 3
            query_emb = embeddings[:batch_size]
            pos_emb = embeddings[batch_size:2*batch_size]
            neg_emb = embeddings[2*batch_size:]
            
            all_query_embeddings.append(query_emb)
            # Combine positive and negative as documents
            all_doc_embeddings.append(torch.cat([pos_emb, neg_emb], dim=0))
    
    # Concatenate embeddings
    query_embeddings = torch.cat(all_query_embeddings, dim=0)
    doc_embeddings = torch.cat(all_doc_embeddings, dim=0)
    
    # Create dummy relevant documents (first half are positives)
    num_queries = query_embeddings.shape[0]
    num_docs_per_query = doc_embeddings.shape[0] // num_queries
    relevant_docs = [[i] for i in range(num_docs_per_query // 2)] * num_queries
    
    # Compute retrieval metrics
    metrics = EmbeddingMetrics.compute_retrieval_metrics(
        query_embeddings, doc_embeddings, relevant_docs
    )
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate BGE-M3 model')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--sts_data', type=str, help='Path to STS evaluation data')
    parser.add_argument('--mteb_data', type=str, help='Path to MTEB evaluation data')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file for results')
    parser.add_argument('--batch_size', type=int, default=64, help='Evaluation batch size')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config (simplified - would need proper config loading)
    # For now, create a minimal config
    class SimpleConfig:
        def __init__(self):
            self.d_model = 1024
            self.n_heads = 16
            self.n_layers = 24
            self.intermediate_size = 4096
            self.vocab_size = 250002
    
    config = SimpleConfig()
    model = ComposerBGEM3(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    results = {}
    
    # Evaluate on STS if data provided
    if args.sts_data:
        print("Evaluating on STS benchmark...")
        sts_dataloader = create_sts_dataloader(
            args.sts_data, 
            batch_size=args.batch_size,
            shuffle=False
        )
        sts_metrics = evaluate_sts(model, sts_dataloader, device)
        results['sts'] = sts_metrics
        print(f"STS Results: {sts_metrics}")
    
    # Evaluate on MTEB if data provided
    if args.mteb_data:
        print("Evaluating on MTEB tasks...")
        mteb_dataloader = create_mteb_dataloader(
            args.mteb_data,
            batch_size=args.batch_size,
            shuffle=False
        )
        mteb_metrics = evaluate_mteb(model, mteb_dataloader, device)
        results['mteb'] = mteb_metrics
        print(f"MTEB Results: {mteb_metrics}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
