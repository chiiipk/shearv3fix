import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import json
import random

class ContrastiveDataset(Dataset):
    """Contrastive learning dataset for embedding training"""
    
    def __init__(self, data_path: str, tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({
                    'anchor': item['anchor'],
                    'positive': item['positive'],
                    'negative': item.get('negative', [])
                })
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Encode anchor
        anchor_encoded = self.tokenizer(
            item['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode positive
        positive_encoded = self.tokenizer(
            item['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': torch.cat([anchor_encoded['input_ids'], positive_encoded['input_ids']], dim=0).squeeze(1),
            'attention_mask': torch.cat([anchor_encoded['attention_mask'], positive_encoded['attention_mask']], dim=0).squeeze(1),
            'labels': torch.tensor([1], dtype=torch.long)  # Positive pair label
        }

def create_contrastive_dataloader(data_path: str, batch_size: int = 32, tokenizer_name: str = "BAAI/bge-m3",
                                 max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """Create contrastive learning dataloader"""
    dataset = ContrastiveDataset(data_path, tokenizer_name, max_length)
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function that flattens paired tensors into shape [batch*2, seq_len].

        Each item currently contains 'input_ids' and 'attention_mask' shaped [2, seq_len].
        Default PyTorch collation would produce [batch, 2, seq_len] which adds an extra
        dimension that the Transformer backbone does not accept. This flattens the
        first two dims so the model receives [batch*2, seq_len].
        """
        # Concatenate along dim=0 to produce [batch*2, seq_len]
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
        attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)

        # Labels are provided per pair as shape (1,) -> make (batch,) then expand to (batch*2,)
        labels = torch.stack([item['labels'].squeeze() for item in batch])  # [batch]
        labels_expanded = labels.repeat_interleave(2)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_expanded,
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn)
