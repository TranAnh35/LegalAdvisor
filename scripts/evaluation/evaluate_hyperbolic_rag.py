#!/usr/bin/env python3
"""
Evaluate and Compare Euclidean RAG vs Hyperbolic RAG.

This script runs the evaluation dataset (test_pairs.jsonl) against the Retrieval Orchestrator
configured in both baseline Euclidean mode and Dual-Space Hyperbolic mode.
Outputs Recall@K and MRR metrics for comparison.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.retrieval.orchestrator import RetrievalOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "zalo-legal"

def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_evaluation_data() -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    queries_path = PROCESSED_DIR / "queries.jsonl"
    pairs_path = PROCESSED_DIR / "test_pairs.jsonl"
    
    # Load queries
    queries_data = load_jsonl(queries_path)
    # Queries might be stored differently in raw queries vs processed queries.
    # We map whatever holds the id to the text.
    q_mapped = {}
    for q in queries_data:
        q_id = q.get('query_id') or q.get('_id')
        text = q.get('text')
        if q_id and text:
            q_mapped[str(q_id)] = text
            
    # Load pairs
    pairs_data = load_jsonl(pairs_path)
    positives = {}
    for p in pairs_data:
        qid = str(p.get("query_id", ""))
        cid = str(p.get("corpus_id", ""))
        if qid and cid and qid in q_mapped:
            if qid not in positives:
                positives[qid] = set()
            positives[qid].add(cid)
            
    # Combine
    final_queries = {qid: q_mapped[qid] for qid in positives.keys()}
    return final_queries, positives

def calculate_metrics(retrieved_docs: List[Dict], true_corpus_ids: Set[str], top_ks: List[int]) -> Dict[str, float]:
    """Calculate recall and reciprocal rank at different K."""
    metrics = {}
    
    retrieved_ids = []
    for doc in retrieved_docs:
        # service._get_chunk_metadata might return corpus_id, doc_id, act_code, etc.
        # Fallback to reconstructing if necessary
        corpus_id = doc.get("corpus_id", "")
        if not corpus_id:
            # Reconstruct from doc_code and article
            if doc.get("doc_code") and doc.get("article"):
                corpus_id = f"{doc['doc_code']}+{doc['article']}"
        retrieved_ids.append(corpus_id)
        
    for k in top_ks:
        k_slice = retrieved_ids[:k]
        hits = sum(1 for cid in k_slice if cid in true_corpus_ids)
        
        metrics[f"Recall@{k}"] = 1.0 if hits > 0 else 0.0
        
        # RR
        rr = 0.0
        for rank, cid in enumerate(k_slice, 1):
            if cid in true_corpus_ids:
                rr = 1.0 / rank
                break
        metrics[f"MRR@{k}"] = rr
        
    return metrics

def evaluate_orchestrator(orchestrator: RetrievalOrchestrator, queries: Dict[str, str], positives: Dict[str, Set[str]], top_ks: List[int]) -> Dict[str, float]:
    max_k = max(top_ks)
    
    total_metrics = {f"Recall@{k}": 0.0 for k in top_ks}
    total_metrics.update({f"MRR@{k}": 0.0 for k in top_ks})
    
    num_queries = len(queries)
    
    # We use tqdm for progress tracking
    for qid, text in tqdm(queries.items(), total=num_queries):
        true_cids = positives[qid]
        results = orchestrator.retrieve(text, top_k=max_k)
        
        query_metrics = calculate_metrics(results, true_cids, top_ks)
        
        for k, v in query_metrics.items():
            total_metrics[k] += v
            
    # Average
    for k in total_metrics:
        total_metrics[k] /= num_queries
        
    return total_metrics

def main():
    parser = argparse.ArgumentParser("Evaluate Hyperbolic RAG")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries for test (0 for all)")
    args = parser.parse_args()
    
    print("Loading datasets...")
    queries, positives = load_evaluation_data()
    print(f"Loaded {len(queries)} queries with ground truth.")
    
    if args.limit > 0:
        queries = {k: queries[k] for k in list(queries.keys())[:args.limit]}
        print(f"Limited to {args.limit} queries.")
        
    top_ks = [1, 5, 10]
    
    # Initialize Baseline Euclidean
    print("\n--- Evaluating Baseline Euclidean RAG ---")
    os.environ["LEGALADVISOR_USE_HYPERBOLIC"] = "0"
    euclidean_orchestrator = RetrievalOrchestrator(use_gpu=True)
    euc_metrics = evaluate_orchestrator(euclidean_orchestrator, queries, positives, top_ks)
    
    # Initialize Hyperbolic RAG
    print("\n--- Evaluating Hyperbolic RAG ---")
    os.environ["LEGALADVISOR_USE_HYPERBOLIC"] = "1"
    hyp_orchestrator = RetrievalOrchestrator(use_gpu=True)
    hyp_metrics = evaluate_orchestrator(hyp_orchestrator, queries, positives, top_ks)
    
    # Print Comparison
    print("\n" + "="*50)
    print("📋 EVALUATION RESULTS")
    print("="*50)
    print(f"{'Metric':<15} | {'Euclidean':<12} | {'Hyperbolic':<12} | {'Improv.'}")
    print("-" * 50)
    
    for metric in sorted(euc_metrics.keys()):
        euc_val = euc_metrics[metric]
        hyp_val = hyp_metrics[metric]
        improv = ((hyp_val - euc_val) / euc_val * 100) if euc_val > 0 else 0.0
        print(f"{metric:<15} | {euc_val:12.4f} | {hyp_val:12.4f} | {improv:+6.2f}%")

if __name__ == "__main__":
    main()
