import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import json
import random

class MTEBDataset(Dataset):
    """MTEB retrieval dataset for contrastive learning"""
    
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
                    'query': item['query'],
                    'positive': item['positive'],
                    'negative': item.get('negative', [])
                })
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Encode query
        query_encoded = self.tokenizer(
            item['query'],
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
        
        # Encode negative (if available)
        if item['negative']:
            negative = random.choice(item['negative'])
            negative_encoded = self.tokenizer(
                negative,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            # Use another random sample as negative
            neg_idx = random.choice([i for i in range(len(self.data)) if i != idx])
            negative_encoded = self.tokenizer(
                self.data[neg_idx]['query'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Combine for triplet
        input_ids = torch.cat([
            query_encoded['input_ids'], 
            positive_encoded['input_ids'], 
            negative_encoded['input_ids']
        ], dim=0)
        
        attention_mask = torch.cat([
            query_encoded['attention_mask'],
            positive_encoded['attention_mask'],
            negative_encoded['attention_mask']
        ], dim=0)
        
        return {
            'input_ids': input_ids.squeeze(1),
            'attention_mask': attention_mask.squeeze(1),
            'positive_ids': torch.tensor([1], dtype=torch.long)  # Indicator for positive pair
        }

def create_mteb_dataloader(data_path: str, batch_size: int = 32, tokenizer_name: str = "BAAI/bge-m3",
                          max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """Create MTEB dataloader"""
    dataset = MTEBDataset(data_path, tokenizer_name, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def collate_mteb_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for MTEB batch processing"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    positive_ids = torch.stack([item['positive_ids'] for item in batch])
    
    # Reshape for triplet processing (query, positive, negative)
    batch_size = input_ids.shape[0]
    input_ids = input_ids.view(batch_size * 3, -1)
    attention_mask = attention_mask.view(batch_size * 3, -1)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'positive_ids': positive_ids.squeeze()
    }
