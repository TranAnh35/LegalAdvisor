#!/usr/bin/env python3
"""
Build Poincaré embeddings index (numpy storage)

Runs the trained HyperbolicEncoder on all chunks to generate coordinates in the Poincare Ball.
Saves the geometries to models/retrieval/index_hyperbolic/embeddings.npy
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

from src.retrieval.hyperbolic.encoder import HyperbolicEncoder
from src.utils.paths import get_processed_data_dir, get_models_retrieval_dir

def build_hyperbolic_index():
    processed_dir = get_processed_data_dir()
    models_dir = get_models_retrieval_dir()
    
    hyperbolic_model_dir = models_dir / "hyperbolic_encoder"
    hyperbolic_index_dir = models_dir / "index_hyperbolic"
    hyperbolic_index_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the model is ready
    if not (hyperbolic_model_dir / "hyperbolic_model.pt").exists():
        print("Hyperbolic encoder not trained yet. Run train_hyperbolic.py first.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base Euclidean model to use as intermediate
    base_encoder_path = models_dir / "zalo_v1"
    base_model = SentenceTransformer(str(base_encoder_path), device="cpu")
    base_dim = base_model.get_sentence_embedding_dimension()
    
    # Initialize Projection Head
    hyperbolic_proj = HyperbolicEncoder(input_dim=base_dim, curvature=1.0)
    
    # Load Weights
    checkpoint = torch.load(hyperbolic_model_dir / "hyperbolic_model.pt", map_location="cpu", weights_only=True)
    if 'poincare_proj' in checkpoint:
        hyperbolic_proj.load_state_dict(checkpoint['poincare_proj'], strict=False)
    hyperbolic_proj.to(device)
    hyperbolic_proj.eval()
    
    # Read Corpus
    chunks_path = processed_dir / "zalo-legal" / "chunks_schema.jsonl"
    print(f"Reading corpus {chunks_path}...")
    
    chunk_ids = []
    chunk_texts = []
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            record = json.loads(line)
            chunk_ids.append(record.get('chunk_id'))
            t = str(record.get('content', '')).replace('_', ' ').strip()
            chunk_texts.append(t)
            
    print(f"Found {len(chunk_ids)} chunks.")
    
    BATCH_SIZE = 128
    embeddings = []
    
    print("Projecting chunks to Hyperbolic Space...")
    for i in tqdm(range(0, len(chunk_texts), BATCH_SIZE), desc="Encoding"):
        batch_texts = chunk_texts[i:i+BATCH_SIZE]
        
        # 1. Euclidean Encode
        euclidean_batch = base_model.encode(batch_texts, convert_to_numpy=False, convert_to_tensor=True, device=device)
        
        # 2. Hyperbolic Map
        with torch.no_grad():
            outputs = hyperbolic_proj(euclidean_batch)
            hyperbolic_batch = outputs["hyperbolic_emb"]
            
        embeddings.append(hyperbolic_batch.cpu().numpy())
        
    embeddings_np = np.vstack(embeddings)
    print(f"Index matrix shape: {embeddings_np.shape}")
    
    # Save Numpy Index
    npy_path = hyperbolic_index_dir / "embeddings.npy"
    np.save(str(npy_path), embeddings_np)
    
    # Save ID map
    id_map = {idx: cid for idx, cid in enumerate(chunk_ids)}
    with open(hyperbolic_index_dir / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f)
        
    print(f"Hyperbolic index saved to {hyperbolic_index_dir}")

if __name__ == "__main__":
    build_hyperbolic_index()
