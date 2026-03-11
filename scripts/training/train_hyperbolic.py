#!/usr/bin/env python3
"""
HyperbolicRAG Training Script

Trains the Poincare projection and depth predictor using a Margin-based Contrastive Loss.
Uses Document -> Article hierarchy from phase 1.
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import geoopt
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.retrieval.hyperbolic.encoder import HyperbolicEncoder
from src.retrieval.hyperbolic.loss import HierarchicalContrastiveLoss
from src.utils.paths import get_processed_data_dir, get_models_retrieval_dir

class LegalHierarchyDataset(Dataset):
    def __init__(self, chunks_path: Path, hierarchy_path: Path):
        self.chunk_texts = {}
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                cid = record.get('chunk_id')
                if cid is not None:
                    self.chunk_texts[str(cid)] = record.get('content', '')
                    
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            self.hierarchy = json.load(f)
            
        self.documents = list(self.hierarchy["documents"].keys())
        # Cache chunk IDs
        self.all_chunk_ids = list(self.hierarchy["chunks"].keys())
        
    def __len__(self):
        return len(self.documents)
        
    def __getitem__(self, idx):
        # Anchor: Parent document
        # But wait, parent document text is just all articles combined or title?
        # A simpler proxy for Document embedding is to just use a randomly sampled Article
        # Let's take doc_code as parent text if no summary exists. Let's assume Document title
        doc_code = self.documents[idx]
        
        # Children articles
        children = self.hierarchy["documents"][doc_code]["chunks"]
        if not children:
            # Fallback
            return {"parent_text": "text", "pos_child_text": "text", "neg_child_text": "text"}
            
        import random
        pos_id = str(random.choice(children))
        pos_text = self.chunk_texts.get(pos_id, "")
        
        # Neg child
        neg_id = str(random.choice(self.all_chunk_ids))
        while neg_id in [str(c) for c in children]:
            neg_id = str(random.choice(self.all_chunk_ids))
            
        neg_text = self.chunk_texts.get(neg_id, "")
        
        # Parent text 
        parent_text = f"Văn bản {doc_code}" # Simple representation
        
        return {
            "parent_text": parent_text,
            "pos_child_text": pos_text,
            "neg_child_text": neg_text
        }

def collate_fn(batch):
    return {
        "parent_text": [b["parent_text"] for b in batch],
        "pos_child_text": [b["pos_child_text"] for b in batch],
        "neg_child_text": [b["neg_child_text"] for b in batch]
    }

def train():
    processed_dir = get_processed_data_dir()
    models_dir = get_models_retrieval_dir()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # 1. Base Encoder
    base_model_path = models_dir / "zalo_v1"
    base_model = SentenceTransformer(str(base_model_path), device="cpu")
    base_dim = base_model.get_sentence_embedding_dimension()
    
    # 2. Hyperbolic Components
    hyperbolic_proj = HyperbolicEncoder(input_dim=base_dim, curvature=1.0).to(device)
    criterion = HierarchicalContrastiveLoss(curvature=1.0, margin=0.1)
    
    # 3. Optimizers
    # RiemannianAdam for parameters inside the Poincare Ball (not strictly needed since
    # our encoder outputs onto the manifold, but safe for geoopt parameters if any)
    optimizer = geoopt.optim.RiemannianAdam(hyperbolic_proj.parameters(), lr=1e-3)
    
    # 4. Data
    dataset = LegalHierarchyDataset(
        chunks_path=processed_dir / "zalo-legal" / "chunks_schema.jsonl",
        hierarchy_path=processed_dir / "zalo-legal" / "hierarchy.json"
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # 5. Training Loop
    EPOCHS = 3
    for epoch in range(EPOCHS):
        hyperbolic_proj.train()
        total_loss = 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress:
            # Get Euclidean Base Embeddings
            with torch.no_grad():
                p_base = base_model.encode(batch["parent_text"], convert_to_tensor=True, device=device)
                pos_base = base_model.encode(batch["pos_child_text"], convert_to_tensor=True, device=device)
                neg_base = base_model.encode(batch["neg_child_text"], convert_to_tensor=True, device=device)
                
            optimizer.zero_grad()
            
            # Forward projection to Poincare
            p_hyp = hyperbolic_proj(p_base)["hyperbolic_emb"]
            pos_hyp = hyperbolic_proj(pos_base)["hyperbolic_emb"]
            neg_hyp = hyperbolic_proj(neg_base)["hyperbolic_emb"]
            
            # Loss
            loss = criterion(p_hyp, pos_hyp, neg_hyp)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(loader):.4f}")
        
    # Save Model
    out_dir = models_dir / "hyperbolic_encoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'poincare_proj': hyperbolic_proj.state_dict()
    }, out_dir / "hyperbolic_model.pt")
    
    print(f"Hyperbolic Model saved to {out_dir}")

if __name__ == "__main__":
    train()
