import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List, Optional
import random

from .sts_dataloader import STSDataset, collate_sts_batch
from .mteb_dataloader import MTEBDataset, collate_mteb_batch

class MixedDataset:
    """Mixed dataset combining STS and MTEB data"""
    
    def __init__(self, sts_path: str, mteb_path: str, sts_ratio: float = 0.5, 
                 tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512):
        self.sts_dataset = STSDataset(sts_path, tokenizer_name, max_length)
        self.mteb_dataset = MTEBDataset(mteb_path, tokenizer_name, max_length)
        self.sts_ratio = sts_ratio
        
        # Calculate dataset sizes based on ratio
        total_size = len(self.sts_dataset) + len(self.mteb_dataset)
        self.sts_size = int(total_size * sts_ratio)
        self.mteb_size = total_size - self.sts_size
    
    def __len__(self) -> int:
        return self.sts_size + self.mteb_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.sts_size:
            # Sample from STS dataset
            sts_idx = idx % len(self.sts_dataset)
            return self.sts_dataset[sts_idx]
        else:
            # Sample from MTEB dataset
            mteb_idx = (idx - self.sts_size) % len(self.mteb_dataset)
            return self.mteb_dataset[mteb_idx]

def create_mixed_dataloader(sts_path: str, mteb_path: str, batch_size: int = 32, 
                           sts_ratio: float = 0.5, tokenizer_name: str = "BAAI/bge-m3",
                           max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """Create mixed dataloader with both STS and MTEB data"""
    dataset = MixedDataset(sts_path, mteb_path, sts_ratio, tokenizer_name, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def collate_mixed_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for mixed batch processing"""
    # Separate STS and MTEB samples
    sts_samples = [item for item in batch if 'similarity_scores' in item]
    mteb_samples = [item for item in batch if 'positive_ids' in item]
    
    result = {}
    
    if sts_samples:
        sts_batch = collate_sts_batch(sts_samples)
        result.update({f"sts_{k}": v for k, v in sts_batch.items()})
    
    if mteb_samples:
        mteb_batch = collate_mteb_batch(mteb_samples)
        result.update({f"mteb_{k}": v for k, v in mteb_batch.items()})
    
    # Combine input_ids and attention_mask if both exist
    if 'sts_input_ids' in result and 'mteb_input_ids' in result:
        result['input_ids'] = torch.cat([result['sts_input_ids'], result['mteb_input_ids']], dim=0)
        result['attention_mask'] = torch.cat([result['sts_attention_mask'], result['mteb_attention_mask']], dim=0)
    elif 'sts_input_ids' in result:
        result['input_ids'] = result['sts_input_ids']
        result['attention_mask'] = result['sts_attention_mask']
    elif 'mteb_input_ids' in result:
        result['input_ids'] = result['mteb_input_ids']
        result['attention_mask'] = result['mteb_attention_mask']
    
    return result
