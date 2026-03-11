"""
Dual-Space Hyperbolic Retrieval Service

This service wraps the standard Euclidean RetrievalService and enhances it 
with a secondary Hyperbolic branch.

Pipeline matches HyperbolicRAG:
Query -> Euclidean -> FAISS L2 -> Euclidean Ranked List
      -> Hyperbolic -> Poincare Numpy -> Hyperbolic Ranked List
Mutual Ranking Fusion -> Final Results
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any

from ..utils.logger import get_logger
from ..utils.paths import get_models_retrieval_dir
from .service import RetrievalService
from .hyperbolic.encoder import HyperbolicEncoder
from .hyperbolic.search import search_poincare_space
from .hyperbolic.fusion import mutual_ranking_fusion

class HyperbolicRetrievalService:
    def __init__(self, use_gpu: bool = False):
        self._logger = get_logger("legaladvisor.hyperbolic_service")
        
        # 1. Initialize Baseline Euclidean Branch
        self.euclidean_service = RetrievalService(use_gpu=use_gpu)
        
        # 2. Hyperbolic Components
        models_dir = get_models_retrieval_dir()
        hyp_model_dir = models_dir / "hyperbolic_encoder"
        hyp_index_dir = models_dir / "index_hyperbolic"
        
        self.hyperbolic_proj = None
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.embeddings = None
        self.id_map = None
        
        self._load_hyperbolic_branch(hyp_model_dir, hyp_index_dir)

    def _load_hyperbolic_branch(self, model_dir: Path, index_dir: Path):
        model_path = model_dir / "hyperbolic_model.pt"
        index_path = index_dir / "embeddings.npy"
        map_path = index_dir / "id_map.json"
        
        if not model_path.exists() or not index_path.exists():
            self._logger.warning("Hyperbolic model/index not found. Will fallback to Euclidean.")
            return
            
        try:
            # Encoder
            base_dim = self.euclidean_service.encoder.get_sentence_embedding_dimension()
            self.hyperbolic_proj = HyperbolicEncoder(input_dim=base_dim, curvature=1.0)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            if 'poincare_proj' in checkpoint:
                self.hyperbolic_proj.load_state_dict(checkpoint['poincare_proj'], strict=False)
            self.hyperbolic_proj.to(self.device).eval()
            
            # Embeddings Pool
            self.embeddings = np.load(str(index_path))
            with open(map_path, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
                
            self._logger.info(f"Loaded {len(self.id_map)} hyperbolic vectors correctly.")
        except Exception as e:
            self._logger.warning(f"Failed to load hyperbolic branch: {e}")
            self.hyperbolic_proj = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Euclidean Branch
        # Use an over-sampled top_k for Euclidean to give fusion a broader pool
        euclidean_results = self.euclidean_service.retrieve(query, top_k=max(top_k * 3, 15))
        
        # Fallback if hyperbolic isn't trained
        if self.hyperbolic_proj is None or self.embeddings is None:
            return euclidean_results[:top_k]
            
        # 2. Hyperbolic Branch
        try:
            # Encode base euclidean directly using underlying model
            euclidean_query = self.euclidean_service.encode_query(query)
            
            # Project to Poincare Ball
            with torch.no_grad():
                q_tensor = torch.tensor(euclidean_query, dtype=torch.float32).to(self.device)
                hyp_tensor = self.hyperbolic_proj(q_tensor)["hyperbolic_emb"]
                hyp_query = hyp_tensor.cpu().numpy()
                
            # Compute distances in numpy
            top_dists, top_idxs = search_poincare_space(
                hyp_query, 
                self.embeddings, 
                top_k=max(top_k * 3, 15), 
                c=1.0
            )
            
            # Map Indices back to actual metadata docs
            hyperbolic_results = []
            for d, idx in zip(top_dists, top_idxs):
                chunk_id_str = self.id_map.get(str(idx), None)
                if chunk_id_str is None:
                    continue
                chunk_id = int(chunk_id_str)
                
                # Use Euclidean service cached metadata routines
                meta = self.euclidean_service._get_chunk_metadata(chunk_id)
                hyp_doc = meta.copy()
                hyp_doc["score"] = float(-d) # Convert distance to score logic
                hyp_doc["chunk_id"] = chunk_id
                hyperbolic_results.append(hyp_doc)
                
            # 3. Dual Space Fusion
            fused_results = mutual_ranking_fusion(
                euclidean_results, 
                hyperbolic_results, 
                top_k=top_k
            )
            
            return fused_results
            
        except Exception as e:
            self._logger.warning(f"Hyperbolic branch failed on query '{query}': {e}")
            return euclidean_results[:top_k]
