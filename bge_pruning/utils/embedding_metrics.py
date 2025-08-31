import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, pearsonr
import torch.nn.functional as F

class EmbeddingMetrics:
    """Metrics computation for embedding models"""
    
    @staticmethod
    def compute_sts_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute STS benchmark metrics"""
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Spearman correlation
        spearman_corr, _ = spearmanr(pred_np, target_np)
        
        # Pearson correlation
        pearson_corr, _ = pearsonr(pred_np, target_np)
        
        # MSE
        mse = np.mean((pred_np - target_np) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_np - target_np))
        
        return {
            'spearman_correlation': spearman_corr,
            'pearson_correlation': pearson_corr,
            'mse': mse,
            'mae': mae
        }
    
    @staticmethod
    def compute_retrieval_metrics(query_embeddings: torch.Tensor, 
                                 doc_embeddings: torch.Tensor,
                                 relevant_docs: List[List[int]],
                                 k_values: List[int] = [1, 5, 10, 100]) -> Dict[str, float]:
        """Compute retrieval metrics (nDCG, MRR, Recall)"""
        batch_size = query_embeddings.shape[0]
        metrics = {f'ndcg_at_{k}': 0.0 for k in k_values}
        metrics.update({f'mrr_at_{k}': 0.0 for k in k_values})
        metrics.update({f'recall_at_{k}': 0.0 for k in k_values})
        
        for i in range(batch_size):
            query_emb = query_embeddings[i:i+1]  # [1, dim]
            
            # Compute similarities
            similarities = F.cosine_similarity(query_emb, doc_embeddings, dim=1)
            
            # Get ranked indices
            _, ranked_indices = torch.sort(similarities, descending=True)
            ranked_indices = ranked_indices.cpu().numpy()
            
            relevant = set(relevant_docs[i]) if i < len(relevant_docs) else set()
            
            for k in k_values:
                top_k = ranked_indices[:k]
                
                # nDCG@k
                ndcg = EmbeddingMetrics._compute_ndcg(top_k, relevant, k)
                metrics[f'ndcg_at_{k}'] += ndcg
                
                # MRR@k
                mrr = EmbeddingMetrics._compute_mrr(top_k, relevant)
                metrics[f'mrr_at_{k}'] += mrr
                
                # Recall@k
                recall = EmbeddingMetrics._compute_recall(top_k, relevant)
                metrics[f'recall_at_{k}'] += recall
        
        # Average across batch
        for key in metrics:
            metrics[key] /= batch_size
        
        return metrics
    
    @staticmethod
    def _compute_ndcg(ranked_list: np.ndarray, relevant_set: set, k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain"""
        if not relevant_set:
            return 0.0
        
        dcg = 0.0
        for i, doc_id in enumerate(ranked_list[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def _compute_mrr(ranked_list: np.ndarray, relevant_set: set) -> float:
        """Compute Mean Reciprocal Rank"""
        for i, doc_id in enumerate(ranked_list):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def _compute_recall(ranked_list: np.ndarray, relevant_set: set) -> float:
        """Compute Recall@k"""
        if not relevant_set:
            return 0.0
        
        retrieved_relevant = len(set(ranked_list) & relevant_set)
        return retrieved_relevant / len(relevant_set)
    
    @staticmethod
    def compute_embedding_similarity_distribution(embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about embedding similarity distribution"""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise similarities
        similarities = torch.mm(embeddings, embeddings.t())
        
        # Remove diagonal (self-similarities)
        mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
        off_diagonal_sims = similarities[mask]
        
        return {
            'similarity_mean': off_diagonal_sims.mean().item(),
            'similarity_std': off_diagonal_sims.std().item(),
            'similarity_min': off_diagonal_sims.min().item(),
            'similarity_max': off_diagonal_sims.max().item(),
            'similarity_median': off_diagonal_sims.median().item()
        }
    
    @staticmethod
    def compute_embedding_diversity(embeddings: torch.Tensor) -> float:
        """Compute embedding diversity (average pairwise distance)"""
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise cosine distances
        similarities = torch.mm(embeddings, embeddings.t())
        distances = 1 - similarities
        
        # Remove diagonal and compute mean
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
        avg_distance = distances[mask].mean().item()
        
        return avg_distance
    
    @staticmethod
    def compute_dimension_utilization(embeddings: torch.Tensor, threshold: float = 0.01) -> Dict[str, float]:
        """Compute how well embedding dimensions are utilized"""
        # Compute standard deviation across samples for each dimension
        dim_stds = embeddings.std(dim=0)
        
        # Count active dimensions (std > threshold)
        active_dims = (dim_stds > threshold).sum().item()
        total_dims = embeddings.shape[1]
        
        return {
            'active_dimensions': active_dims,
            'total_dimensions': total_dims,
            'dimension_utilization': active_dims / total_dims,
            'avg_dimension_std': dim_stds.mean().item(),
            'min_dimension_std': dim_stds.min().item(),
            'max_dimension_std': dim_stds.max().item()
        }
