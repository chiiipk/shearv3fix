import os
import sys
import json
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from bge_pruning.data_modules.contrastive_dataset import create_contrastive_dataloader

# Create small test JSONL data
data_path = repo_root / 'scripts' / 'test_data.jsonl'
entries = [
    {'anchor': 'The cat sits on the mat.', 'positive': 'A cat is sitting on a mat.'},
    {'anchor': 'An apple a day keeps the doctor away.', 'positive': 'Eating an apple each day keeps doctors away.'}
]
with open(data_path, 'w', encoding='utf-8') as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')

# Create dataloader
loader = create_contrastive_dataloader(str(data_path), batch_size=2, shuffle=False)

for batch in loader:
    print('input_ids.shape =', batch['input_ids'].shape)
    print('attention_mask.shape =', batch['attention_mask'].shape)
    print('labels.shape =', batch['labels'].shape)
    print('labels =', batch['labels'])
    break
