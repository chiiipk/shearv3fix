import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import json
import csv

class STSDataset(Dataset):
    """STS benchmark dataset for sentence similarity tasks"""
    
    def __init__(self, data_path: str, tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 3:
                    score = float(row[0]) if row[0].replace('.', '').isdigit() else 0.0
                    sent1 = row[1]
                    sent2 = row[2]
                    data.append({
                        'sentence1': sent1,
                        'sentence2': sent2,
                        'score': score
                    })
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize both sentences
        encoded1 = self.tokenizer(
            item['sentence1'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoded2 = self.tokenizer(
            item['sentence2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Combine for batch processing
        input_ids = torch.cat([encoded1['input_ids'], encoded2['input_ids']], dim=0)
        attention_mask = torch.cat([encoded1['attention_mask'], encoded2['attention_mask']], dim=0)
        
        return {
            'input_ids': input_ids.squeeze(1),
            'attention_mask': attention_mask.squeeze(1),
            'similarity_scores': torch.tensor(item['score'], dtype=torch.float)
        }

def create_sts_dataloader(data_path: str, batch_size: int = 32, tokenizer_name: str = "BAAI/bge-m3", 
                         max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """Create STS dataloader"""
    dataset = STSDataset(data_path, tokenizer_name, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def collate_sts_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for STS batch processing"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    similarity_scores = torch.stack([item['similarity_scores'] for item in batch])
    
    # Reshape for sentence pair processing
    batch_size = input_ids.shape[0]
    input_ids = input_ids.view(batch_size * 2, -1)
    attention_mask = attention_mask.view(batch_size * 2, -1)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'similarity_scores': similarity_scores
    }
