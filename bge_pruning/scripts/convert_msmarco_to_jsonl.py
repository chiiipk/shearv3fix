#!/usr/bin/env python3
"""
Convert MS MARCO dataset to JSONL format for BGE-M3 pruning training
Processes official MS MARCO files into the format expected by MTEB dataloader
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_corpus(corpus_path: str) -> Dict[str, str]:
    """Load MS MARCO corpus"""
    corpus = {}
    logger.info(f"Loading corpus from {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                doc_id, text = row[0], row[1]
                corpus[doc_id] = text
    
    logger.info(f"Loaded {len(corpus)} documents")
    return corpus

def load_queries(queries_path: str) -> Dict[str, str]:
    """Load MS MARCO queries"""
    queries = {}
    logger.info(f"Loading queries from {queries_path}")
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                query_id, text = row[0], row[1]
                queries[query_id] = text
    
    logger.info(f"Loaded {len(queries)} queries")
    return queries

def load_qrels(qrels_path: str) -> Dict[str, Set[str]]:
    """Load MS MARCO relevance judgments"""
    qrels = {}
    logger.info(f"Loading qrels from {qrels_path}")
    
    with open(qrels_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 4:
                query_id, _, doc_id, relevance = row
                if int(relevance) > 0:  # Only positive examples
                    if query_id not in qrels:
                        qrels[query_id] = set()
                    qrels[query_id].add(doc_id)
    
    total_pairs = sum(len(docs) for docs in qrels.values())
    logger.info(f"Loaded {len(qrels)} queries with {total_pairs} relevant documents")
    return qrels

def convert_to_jsonl(queries: Dict[str, str], corpus: Dict[str, str], 
                    qrels: Dict[str, Set[str]], output_path: str, max_examples: int = None):
    """Convert to JSONL format for MTEB training"""
    logger.info(f"Converting to JSONL format: {output_path}")
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries.items():
            if query_id not in qrels:
                continue
                
            for doc_id in qrels[query_id]:
                if doc_id not in corpus:
                    continue
                
                # Create training example
                example = {
                    "query": query_text,
                    "document": corpus[doc_id],
                    "query_id": query_id,
                    "document_id": doc_id,
                    "label": 1  # Positive example
                }
                
                f.write(json.dumps(example) + '\n')
                count += 1
                
                if max_examples and count >= max_examples:
                    break
            
            if max_examples and count >= max_examples:
                break
    
    logger.info(f"Wrote {count} training examples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert MS MARCO to JSONL for BGE-M3 training')
    parser.add_argument('--queries', required=True, help='Path to queries.train.tsv')
    parser.add_argument('--corpus', required=True, help='Path to corpus.tsv')
    parser.add_argument('--qrels', required=True, help='Path to qrels.train.tsv')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--max_examples', type=int, help='Maximum number of examples to generate')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.queries, args.corpus, args.qrels]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return 1
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load all data
        corpus = load_corpus(args.corpus)
        queries = load_queries(args.queries)
        qrels = load_qrels(args.qrels)
        
        # Convert to JSONL
        convert_to_jsonl(queries, corpus, qrels, args.output, args.max_examples)
        
        logger.info("Conversion completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
