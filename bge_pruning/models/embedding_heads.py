import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class DenseEmbeddingHead(nn.Module):
    """Dense embedding head for BGE-M3"""
    
    def __init__(self, hidden_size: int, output_dim: int = None):
        super().__init__()
        self.output_dim = output_dim or hidden_size
        self.dense = nn.Linear(hidden_size, self.output_dim)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pool the hidden states (use CLS token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Use CLS token
            pooled = hidden_states[:, 0]
        
        dense_embedding = self.dense(pooled)
        dense_embedding = self.activation(dense_embedding)
        return F.normalize(dense_embedding, p=2, dim=-1)

class SparseEmbeddingHead(nn.Module):
    """Sparse embedding head for BGE-M3"""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.vocab_size = vocab_size
        
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute token importance weights
        token_weights = self.linear(hidden_states).squeeze(-1)
        
        if attention_mask is not None:
            token_weights = token_weights * attention_mask
        
        # Apply ReLU and normalize
        token_weights = F.relu(token_weights)
        
        # Create sparse representation
        batch_size, seq_len = input_ids.shape
        sparse_embedding = torch.zeros(batch_size, self.vocab_size, device=input_ids.device)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if attention_mask is None or attention_mask[i, j] > 0:
                    token_id = input_ids[i, j].item()
                    if 0 <= token_id < self.vocab_size:
                        sparse_embedding[i, token_id] += token_weights[i, j]
        
        return sparse_embedding

class MultiVectorEmbeddingHead(nn.Module):
    """Multi-vector embedding head for BGE-M3"""
    
    def __init__(self, hidden_size: int, output_dim: int = None):
        super().__init__()
        self.output_dim = output_dim or hidden_size
        self.dense = nn.Linear(hidden_size, self.output_dim)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Transform all token representations
        multi_vector = self.dense(hidden_states)
        
        if attention_mask is not None:
            # Zero out padded tokens
            multi_vector = multi_vector * attention_mask.unsqueeze(-1)
        
        # Normalize each vector
        multi_vector = F.normalize(multi_vector, p=2, dim=-1)
        return multi_vector

class BGEEmbeddingHeads(nn.Module):
    """Combined embedding heads for BGE-M3 model"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = getattr(config, 'vocab_size', 250002)
        
        # Initialize all three heads
        self.dense_head = DenseEmbeddingHead(self.hidden_size)
        self.sparse_head = SparseEmbeddingHead(self.hidden_size, self.vocab_size)
        self.multi_vector_head = MultiVectorEmbeddingHead(self.hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dense: bool = True,
                return_sparse: bool = True,
                return_multi_vector: bool = True) -> Dict[str, torch.Tensor]:
        
        outputs = {}
        
        if return_dense:
            outputs['dense_embedding'] = self.dense_head(hidden_states, attention_mask)
        
        if return_sparse:
            outputs['sparse_embedding'] = self.sparse_head(hidden_states, input_ids, attention_mask)
        
        if return_multi_vector:
            outputs['multi_vector_embedding'] = self.multi_vector_head(hidden_states, attention_mask)
        
        return outputs
    
    def compute_similarity(self, 
                          query_embeddings: Dict[str, torch.Tensor],
                          doc_embeddings: Dict[str, torch.Tensor],
                          weights: Dict[str, float] = None) -> torch.Tensor:
        """Compute similarity scores between query and document embeddings"""
        
        if weights is None:
            weights = {'dense': 1.0, 'sparse': 0.3, 'multi_vector': 1.0}
        
        similarities = []
        
        # Dense similarity
        if 'dense_embedding' in query_embeddings and 'dense_embedding' in doc_embeddings:
            dense_sim = torch.mm(query_embeddings['dense_embedding'], 
                               doc_embeddings['dense_embedding'].t())
            similarities.append(weights.get('dense', 1.0) * dense_sim)
        
        # Sparse similarity
        if 'sparse_embedding' in query_embeddings and 'sparse_embedding' in doc_embeddings:
            sparse_sim = torch.mm(query_embeddings['sparse_embedding'],
                                doc_embeddings['sparse_embedding'].t())
            similarities.append(weights.get('sparse', 0.3) * sparse_sim)
        
        # Multi-vector similarity (max pooling)
        if 'multi_vector_embedding' in query_embeddings and 'multi_vector_embedding' in doc_embeddings:
            q_mv = query_embeddings['multi_vector_embedding']  # [batch_q, seq_len, dim]
            d_mv = doc_embeddings['multi_vector_embedding']    # [batch_d, seq_len, dim]
            
            # Compute all pairwise similarities and take max
            mv_sim = torch.einsum('qsd,dtd->qst', q_mv, d_mv.transpose(0, 1))
            mv_sim = mv_sim.max(dim=-1)[0].max(dim=-1)[0]  # Max over both sequence dimensions
            similarities.append(weights.get('multi_vector', 1.0) * mv_sim)
        
        # Combine similarities
        if similarities:
            total_similarity = torch.stack(similarities, dim=0).sum(dim=0)
        else:
            raise ValueError("No valid embedding pairs found for similarity computation")
        
        return total_similarity
    
    def prune_params(self, hidden_z: Optional[torch.Tensor] = None):
        """Prune embedding head parameters based on hidden dimension mask"""
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            
            # Prune dense head
            if hasattr(self.dense_head.dense, 'weight'):
                self.dense_head.dense.weight = nn.Parameter(
                    self.dense_head.dense.weight.data[:, remaining_index]
                )
            
            # Prune sparse head
            if hasattr(self.sparse_head.linear, 'weight'):
                self.sparse_head.linear.weight = nn.Parameter(
                    self.sparse_head.linear.weight.data[:, remaining_index]
                )
            
            # Prune multi-vector head
            if hasattr(self.multi_vector_head.dense, 'weight'):
                self.multi_vector_head.dense.weight = nn.Parameter(
                    self.multi_vector_head.dense.weight.data[:, remaining_index]
                )
