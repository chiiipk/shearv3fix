"""
Production BGE-M3 Dataset Loader with MTEB Support
Clean implementation for STS similarity and MTEB retrieval tasks
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import datasets
from typing import Dict, Any

class BGEDataset(torch.utils.data.Dataset):
    """Production dataset for BGE-M3 training with proper tensor handling"""
    
    def __init__(self, dataset_name: str, split: str = "train", 
                 tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Production dataset loading
        if dataset_name == "sts":
            self.dataset = datasets.load_dataset("mteb/stsbenchmark-sts", split=split)
            self.task_type = "similarity"
        elif dataset_name == "msmarco":
            self.dataset = datasets.load_dataset("ms_marco", "v2.1", split=split)
            self.task_type = "retrieval"
        elif dataset_name.startswith("mteb/"):
            # MTEB benchmark datasets
            self.dataset = datasets.load_dataset(dataset_name, split=split)
            self.task_type = "retrieval"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: sts, msmarco, mteb/*")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        if self.task_type == "similarity":
            # STS: Clean sentence pair processing
            inputs1 = self.tokenizer(
                item['sentence1'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            inputs2 = self.tokenizer(
                item['sentence2'], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            
            return {
                'input_ids_1': inputs1['input_ids'].squeeze(0),  # [seq_length]
                'attention_mask_1': inputs1['attention_mask'].squeeze(0),  # [seq_length]
                'input_ids_2': inputs2['input_ids'].squeeze(0),  # [seq_length]
                'attention_mask_2': inputs2['attention_mask'].squeeze(0),  # [seq_length]
                'similarity_score': torch.tensor(item['score'], dtype=torch.float),
                'task_type': 'sts'
            }
        
        else:  # retrieval
            # MTEB/MSMARCO: Clean query-passage processing
            query_key = 'query' if 'query' in item else 'question'
            passage_key = 'passage' if 'passage' in item else 'text' if 'text' in item else 'positive_passages'
            
            query_inputs = self.tokenizer(
                item[query_key], max_length=self.max_length,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            passage_inputs = self.tokenizer(
                item[passage_key][0] if isinstance(item[passage_key], list) else item[passage_key], 
                max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
            )
            
            return {
                'input_ids_1': query_inputs['input_ids'].squeeze(0),  # [seq_length]
                'attention_mask_1': query_inputs['attention_mask'].squeeze(0),  # [seq_length]
                'input_ids_2': passage_inputs['input_ids'].squeeze(0),  # [seq_length]
                'attention_mask_2': passage_inputs['attention_mask'].squeeze(0),  # [seq_length]
                'task_type': 'retrieval'
            }

def create_dataloader(dataset_name: str, split: str = "train", batch_size: int = 16, 
                     tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512,
                     num_workers: int = 2) -> DataLoader:
    """Create production DataLoader with proper tensor handling"""
    dataset = BGEDataset(dataset_name, split, tokenizer_name, max_length)
    
    def collate_fn(batch):
        """Production collate function - maintains paired structure"""
        # Stack tensors to create proper batch dimensions
        input_ids_1 = torch.stack([item['input_ids_1'] for item in batch])  # [batch_size, seq_length]
        attention_mask_1 = torch.stack([item['attention_mask_1'] for item in batch])  # [batch_size, seq_length]
        input_ids_2 = torch.stack([item['input_ids_2'] for item in batch])  # [batch_size, seq_length]
        attention_mask_2 = torch.stack([item['attention_mask_2'] for item in batch])  # [batch_size, seq_length]
        
        # Interleave pairs: [sent1_1, sent2_1, sent1_2, sent2_2, ...]
        batch_size = len(batch)
        input_ids = torch.zeros(batch_size * 2, input_ids_1.size(1), dtype=input_ids_1.dtype)
        attention_mask = torch.zeros(batch_size * 2, attention_mask_1.size(1), dtype=attention_mask_1.dtype)
        
        input_ids[0::2] = input_ids_1  # Even indices: first sentences
        input_ids[1::2] = input_ids_2  # Odd indices: second sentences
        attention_mask[0::2] = attention_mask_1
        attention_mask[1::2] = attention_mask_2
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # Add task-specific targets (infer task from data, no string fields for Composer)
        if batch[0]['task_type'] == 'sts':
            similarity_scores = torch.stack([item['similarity_score'] for item in batch])
            result['similarity_scores'] = similarity_scores
        
        return result
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )