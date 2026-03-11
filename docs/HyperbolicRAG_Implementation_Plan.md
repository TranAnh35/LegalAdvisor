# 📋 HyperbolicRAG Implementation Plan

## Tổng quan

**Mục tiêu:** Nâng cấp hệ thống RAG từ Naive RAG sang HyperbolicRAG để capture hierarchical structure của dữ liệu luật pháp Việt Nam.

**Paper tham khảo:** [HyperbolicRAG: Enhancing RAG with Hyperbolic Representations](https://arxiv.org/abs/2511.18808)

---

## 1. Kiến trúc Mới

```
┌─────────────────────────────────────────────────────────────────┐
│                    HyperbolicRAG Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query ──┬──► Euclidean Encoder ──► FAISS Euclidean ──┐         │
│          │                                             │         │
│          └──► Hyperbolic Encoder ──► FAISS Poincaré ──┼──► Fusion│
│                                                        │         │
│                                                        ▼         │
│                                              Mutual-Ranking      │
│                                                   ▼              │
│                                              Top-K Results       │
│                                                   ▼              │
│                                              Gemini LLM          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Các Phase Triển khai

### Phase 1: Chuẩn bị Hierarchy Data ✅ COMPLETED
**Thời gian:** Hoàn thành  
**File output:** 
- `data/analysis/dataset_analysis_report.json` (17KB)
- `data/analysis/id_patterns_analysis.json` (28KB)
- `data/analysis/document_structure.json`
- `data/processed/zalo-legal/hierarchy.json` (26MB, 915K lines)
- `data/processed/zalo-legal/chunk_depths.json` (61,425 mappings)

**Results:**
- 📊 61,425 total records (articles)
- 📁 3,271 unique documents
- 🏛️ 16 issuers (Quốc hội, Chính phủ, các Bộ...)
- 📋 20 document types (Luật, Nghị định, Thông tư...)
- 🌳 Hierarchy: Root → Issuer → DocType → Document → Article

**Tasks:**
- [x] Task 1.1: Analyze dataset (analyze_dataset.py)
- [x] Task 1.2: Extract ID patterns (analyze_id_patterns.py)
- [x] Task 1.3: Build document structure
- [x] Task 1.4: Build hierarchy with depth assignments (build_hierarchy.py)

### Phase 2: Train Hyperbolic Embeddings ✅ COMPLETED
**Thời gian:** 2-3 ngày (compute)  
**File output:** `models/retrieval/hyperbolic_encoder/`

**Tasks:**
- [x] Task 2.1: Implement Poincaré projection layer
- [x] Task 2.2: Implement hierarchical contrastive loss
- [x] Task 2.3: Fine-tune từ E5 model hiện có
- [x] Task 2.4: Evaluate embedding quality

### Phase 3: Build Dual Index ✅ COMPLETED
**Thời gian:** 1 ngày  
**File output:** `models/retrieval/index_hyperbolic/`

**Tasks:**
- [x] Task 3.1: Generate hyperbolic embeddings cho corpus
- [x] Task 3.2: Build Numpy Geometry index cho Poincaré space (Thay FAISS do L2 Error)
- [x] Task 3.3: Create unified id_map

### Phase 4: Implement Fusion Retrieval ✅ COMPLETED
**Thời gian:** 1-2 ngày  
**File output:** `src/retrieval/hyperbolic_service.py`

**Tasks:**
- [x] Task 4.1: Implement HyperbolicRetrievalService & Orchestrator
- [x] Task 4.2: Implement mutual-ranking fusion & Hierarchical Reranker
- [x] Task 4.3: Integrate với GeminiRAG
- [x] Task 4.4: Testing và benchmarking

---

## 3. Chi tiết Triển khai

### 3.1 Phase 1: Hierarchy Data

#### Cấu trúc Hierarchy cho Luật Việt Nam

```python
LEGAL_HIERARCHY = {
    0: "domain",      # Lĩnh vực: Dân sự, Hình sự, Hành chính...
    1: "document",    # Văn bản: Bộ luật Dân sự 2015, BLHS 2015...
    2: "part",        # Phần: Phần I, Phần II...
    3: "chapter",     # Chương: Chương I, Chương II...
    4: "section",     # Mục: Mục 1, Mục 2...
    5: "article",     # Điều: Điều 1, Điều 2...
    6: "clause",      # Khoản: Khoản 1, Khoản 2...
    7: "point",       # Điểm: Điểm a, Điểm b...
}
```

#### Script: `scripts/build_hierarchy.py`

```python
"""
Build hierarchy metadata từ corpus luật.
Output: data/processed/zalo-legal/hierarchy.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

def extract_depth_from_content(content: str, corpus_id: str) -> int:
    """
    Xác định depth level dựa trên content và corpus_id.
    
    Returns:
        int: Depth level (0-7, nhỏ hơn = general hơn)
    """
    # Mặc định: article level (depth=5)
    depth = 5
    
    # Check patterns để xác định depth chính xác hơn
    content_lower = content.lower()
    
    if re.search(r'^điểm\s+[a-zđ][\.\)]', content_lower):
        depth = 7  # Điểm
    elif re.search(r'^khoản\s+\d+', content_lower):
        depth = 6  # Khoản
    elif re.search(r'^điều\s+\d+', content_lower):
        depth = 5  # Điều
    elif re.search(r'^mục\s+\d+', content_lower):
        depth = 4  # Mục
    elif re.search(r'^chương\s+[ivxlcdm\d]+', content_lower):
        depth = 3  # Chương
    elif re.search(r'^phần\s+(thứ\s+)?[ivxlcdm\d]+', content_lower):
        depth = 2  # Phần
        
    return depth


def extract_parent_id(corpus_id: str, depth: int) -> Optional[str]:
    """
    Xác định parent_id dựa trên corpus_id và depth.
    
    Ví dụ:
        corpus_id="blds-2015+125" (Điều 125 BLDS)
        -> parent_id="blds-2015" (Văn bản BLDS 2015)
    """
    if depth <= 1:
        return None  # Document level không có parent
    
    parts = corpus_id.split('+')
    if len(parts) >= 2:
        return parts[0]  # Parent là document
    return None


def build_hierarchy(jsonl_path: Path) -> Dict[str, Any]:
    """
    Build hierarchy từ JSONL corpus.
    
    Returns:
        Dict với structure:
        {
            "chunks": {
                chunk_id: {
                    "corpus_id": str,
                    "depth": int,
                    "parent_id": str,
                    "children_ids": List[str]
                }
            },
            "documents": {
                doc_code: {
                    "chunks": List[int],
                    "depth": 1
                }
            },
            "stats": {...}
        }
    """
    hierarchy = {
        "chunks": {},
        "documents": {},
        "stats": {
            "total_chunks": 0,
            "depth_distribution": {}
        }
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            chunk_id = record.get('chunk_id')
            corpus_id = record.get('corpus_id', '')
            content = record.get('content', '')
            
            if chunk_id is None:
                continue
            
            depth = extract_depth_from_content(content, corpus_id)
            parent_id = extract_parent_id(corpus_id, depth)
            
            hierarchy["chunks"][chunk_id] = {
                "corpus_id": corpus_id,
                "depth": depth,
                "parent_id": parent_id,
                "children_ids": []
            }
            
            # Track document
            doc_code = corpus_id.split('+')[0] if '+' in corpus_id else corpus_id
            if doc_code:
                if doc_code not in hierarchy["documents"]:
                    hierarchy["documents"][doc_code] = {
                        "chunks": [],
                        "depth": 1
                    }
                hierarchy["documents"][doc_code]["chunks"].append(chunk_id)
            
            # Stats
            hierarchy["stats"]["total_chunks"] += 1
            depth_key = str(depth)
            hierarchy["stats"]["depth_distribution"][depth_key] = \
                hierarchy["stats"]["depth_distribution"].get(depth_key, 0) + 1
    
    return hierarchy


def main():
    from src.utils.paths import get_processed_data_dir
    
    processed_dir = get_processed_data_dir()
    jsonl_path = processed_dir / "zalo-legal" / "chunks_schema.jsonl"
    output_path = processed_dir / "zalo-legal" / "hierarchy.json"
    
    print(f"Building hierarchy from: {jsonl_path}")
    hierarchy = build_hierarchy(jsonl_path)
    
    print(f"Stats: {hierarchy['stats']}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)
    
    print(f"Saved hierarchy to: {output_path}")


if __name__ == "__main__":
    main()
```

---

### 3.2 Phase 2: Hyperbolic Embeddings

#### Dependencies cần thêm

```txt
# Thêm vào requirements.txt
geoopt>=0.5.0          # Riemannian optimization
```

#### Script: `scripts/train_hyperbolic.py`

```python
"""
Train Hyperbolic (Poincaré) embeddings cho legal corpus.

Approach:
1. Load pre-trained E5 encoder
2. Add Poincaré projection layer
3. Fine-tune với hierarchical contrastive loss
4. Export model và embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Hyperbolic geometry operations
import geoopt

from sentence_transformers import SentenceTransformer


class PoincareProjection(nn.Module):
    """
    Project Euclidean embeddings vào Poincaré Ball.
    
    Poincaré Ball: B^n = {x ∈ R^n : ||x|| < 1}
    """
    
    def __init__(self, input_dim: int, output_dim: int, curvature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.curvature = curvature
        
        # Linear projection
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Poincaré Ball manifold
        self.ball = geoopt.PoincareBall(c=curvature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project Euclidean vectors vào Poincaré Ball.
        
        Args:
            x: Euclidean embeddings [batch, input_dim]
        
        Returns:
            Poincaré embeddings [batch, output_dim]
        """
        # Linear transform
        h = self.linear(x)
        
        # Exponential map từ origin (0) vào Poincaré Ball
        # Ensures ||output|| < 1
        h = self.ball.expmap0(h)
        
        return h


class HyperbolicEncoder(nn.Module):
    """
    Encoder kết hợp Euclidean + Hyperbolic embeddings.
    """
    
    def __init__(
        self,
        base_encoder: SentenceTransformer,
        hyperbolic_dim: int = 384,
        curvature: float = 1.0
    ):
        super().__init__()
        self.base_encoder = base_encoder
        
        # Freeze base encoder (optional - có thể fine-tune)
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        
        # Get embedding dimension từ base encoder
        euclidean_dim = base_encoder.get_sentence_embedding_dimension()
        
        # Poincaré projection
        self.poincare_proj = PoincareProjection(
            input_dim=euclidean_dim,
            output_dim=hyperbolic_dim,
            curvature=curvature
        )
        
        self.hyperbolic_dim = hyperbolic_dim
    
    def encode_euclidean(self, texts: List[str]) -> torch.Tensor:
        """Encode texts thành Euclidean embeddings."""
        return torch.tensor(
            self.base_encoder.encode(texts, convert_to_numpy=True),
            dtype=torch.float32
        )
    
    def encode_hyperbolic(self, texts: List[str]) -> torch.Tensor:
        """Encode texts thành Poincaré embeddings."""
        euclidean = self.encode_euclidean(texts)
        return self.poincare_proj(euclidean)
    
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both Euclidean and Hyperbolic embeddings.
        """
        euclidean = self.encode_euclidean(texts)
        hyperbolic = self.poincare_proj(euclidean)
        return euclidean, hyperbolic


class HierarchicalContrastiveLoss(nn.Module):
    """
    Loss function cho hierarchical embeddings.
    
    Objectives:
    1. Semantic similarity: similar texts gần nhau
    2. Hierarchical containment: parent gần center hơn children
    """
    
    def __init__(self, ball: geoopt.PoincareBall, margin: float = 0.1):
        super().__init__()
        self.ball = ball
        self.margin = margin
    
    def poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Poincaré distance giữa 2 points."""
        return self.ball.dist(x, y)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        depths: torch.Tensor,
        parent_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hierarchical loss.
        
        Args:
            embeddings: [batch, dim] Poincaré embeddings
            depths: [batch] depth levels (0=root, higher=more specific)
            parent_indices: [batch] index of parent for each sample (-1 if no parent)
        
        Returns:
            Loss scalar
        """
        batch_size = embeddings.size(0)
        
        # 1. Depth ordering loss: deeper nodes should be farther from origin
        norms = torch.norm(embeddings, dim=-1)  # Distance from origin
        
        # Expected: higher depth -> higher norm (farther from center)
        # Normalize depth to [0, 1]
        max_depth = depths.max().float() + 1e-6
        normalized_depths = depths.float() / max_depth
        
        # MSE between norm và expected depth position
        depth_loss = F.mse_loss(norms, normalized_depths * 0.9)  # 0.9 to keep inside ball
        
        # 2. Parent-child ordering loss (if parent_indices provided)
        parent_loss = torch.tensor(0.0, device=embeddings.device)
        if parent_indices is not None:
            valid_mask = parent_indices >= 0
            if valid_mask.any():
                child_emb = embeddings[valid_mask]
                parent_emb = embeddings[parent_indices[valid_mask].long()]
                
                child_norms = torch.norm(child_emb, dim=-1)
                parent_norms = torch.norm(parent_emb, dim=-1)
                
                # Children should be farther from origin than parents
                violations = F.relu(parent_norms - child_norms + self.margin)
                parent_loss = violations.mean()
        
        total_loss = depth_loss + parent_loss
        return total_loss


def train_hyperbolic_encoder(
    base_model_path: str,
    hierarchy_path: Path,
    chunks_path: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    hyperbolic_dim: int = 384
):
    """
    Train hyperbolic encoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base encoder
    print(f"Loading base encoder from: {base_model_path}")
    base_encoder = SentenceTransformer(base_model_path, device="cpu")
    
    # Initialize hyperbolic encoder
    encoder = HyperbolicEncoder(
        base_encoder=base_encoder,
        hyperbolic_dim=hyperbolic_dim
    ).to(device)
    
    # Load hierarchy
    print(f"Loading hierarchy from: {hierarchy_path}")
    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hierarchy = json.load(f)
    
    # Load chunks for text content
    print(f"Loading chunks from: {chunks_path}")
    chunk_texts = {}
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            chunk_id = record.get('chunk_id')
            content = record.get('content', '')
            if chunk_id is not None:
                chunk_texts[chunk_id] = content
    
    # Prepare training data
    chunk_ids = list(hierarchy["chunks"].keys())
    
    # Loss and optimizer
    ball = geoopt.PoincareBall(c=1.0)
    criterion = HierarchicalContrastiveLoss(ball)
    optimizer = geoopt.optim.RiemannianAdam(
        encoder.poincare_proj.parameters(),
        lr=learning_rate
    )
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle
        indices = torch.randperm(len(chunk_ids))
        
        for i in tqdm(range(0, len(chunk_ids), batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[i:i+batch_size]
            batch_chunk_ids = [chunk_ids[j] for j in batch_indices]
            
            # Get texts and hierarchy info
            texts = [chunk_texts.get(cid, "") for cid in batch_chunk_ids]
            depths = torch.tensor([
                hierarchy["chunks"].get(str(cid), {}).get("depth", 5)
                for cid in batch_chunk_ids
            ], device=device)
            
            # Get parent indices (within batch) - simplified
            parent_indices = torch.full((len(batch_chunk_ids),), -1, device=device)
            
            # Forward
            _, hyperbolic_emb = encoder(texts)
            hyperbolic_emb = hyperbolic_emb.to(device)
            
            # Loss
            loss = criterion(hyperbolic_emb, depths, parent_indices)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'poincare_proj': encoder.poincare_proj.state_dict(),
        'hyperbolic_dim': hyperbolic_dim,
        'curvature': 1.0
    }, output_dir / "hyperbolic_encoder.pt")
    
    print(f"Saved model to: {output_dir}")
    
    return encoder


def main():
    from src.utils.paths import get_processed_data_dir, get_models_retrieval_dir
    
    processed_dir = get_processed_data_dir()
    models_dir = get_models_retrieval_dir()
    
    train_hyperbolic_encoder(
        base_model_path=str(models_dir / "zalo_v1"),
        hierarchy_path=processed_dir / "zalo-legal" / "hierarchy.json",
        chunks_path=processed_dir / "zalo-legal" / "chunks_schema.jsonl",
        output_dir=models_dir / "hyperbolic_encoder",
        epochs=5,
        batch_size=32
    )


if __name__ == "__main__":
    main()
```

---

### 3.3 Phase 3: Build Dual Index

#### Script: `scripts/build_hyperbolic_index.py`

```python
"""
Build FAISS index cho Hyperbolic embeddings.
"""

import json
import numpy as np
import faiss
import torch
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


def build_hyperbolic_index(
    encoder_path: Path,
    base_model_path: Path,
    chunks_path: Path,
    output_dir: Path,
    batch_size: int = 64
):
    """
    Generate Poincaré embeddings và build FAISS index.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    print("Loading models...")
    base_encoder = SentenceTransformer(str(base_model_path), device="cpu")
    
    checkpoint = torch.load(encoder_path / "hyperbolic_encoder.pt", map_location="cpu")
    hyperbolic_dim = checkpoint['hyperbolic_dim']
    
    # Rebuild projection layer
    from scripts.train_hyperbolic import PoincareProjection
    poincare_proj = PoincareProjection(
        input_dim=base_encoder.get_sentence_embedding_dimension(),
        output_dim=hyperbolic_dim,
        curvature=checkpoint['curvature']
    )
    poincare_proj.load_state_dict(checkpoint['poincare_proj'])
    poincare_proj.to(device)
    poincare_proj.eval()
    
    # Load chunks
    print("Loading chunks...")
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            chunks.append(record)
    
    # Generate embeddings
    print(f"Generating {len(chunks)} embeddings...")
    all_embeddings = []
    id_map = []
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        texts = [c.get('content', '')[:512] for c in batch]  # Truncate
        
        # Euclidean embeddings
        with torch.no_grad():
            euclidean = torch.tensor(
                base_encoder.encode(texts, convert_to_numpy=True),
                dtype=torch.float32,
                device=device
            )
            
            # Project to Poincaré
            hyperbolic = poincare_proj(euclidean)
            all_embeddings.append(hyperbolic.cpu().numpy())
        
        # ID mapping
        for j, chunk in enumerate(batch):
            id_map.append({
                "faiss_id": i + j,
                "chunk_id": chunk.get('chunk_id'),
                "corpus_id": chunk.get('corpus_id')
            })
    
    # Concatenate embeddings
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Build FAISS index
    # Sử dụng L2 distance vì Poincaré distance ~ L2 trong small regions
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(hyperbolic_dim)
    index.add(embeddings)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_dir / "hyperbolic_index.faiss"))
    
    with open(output_dir / "hyperbolic_id_map.jsonl", 'w', encoding='utf-8') as f:
        for entry in id_map:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Save embeddings for analysis
    np.save(output_dir / "hyperbolic_embeddings.npy", embeddings)
    
    print(f"Saved index to: {output_dir}")
    print(f"Total vectors: {index.ntotal}")


def main():
    from src.utils.paths import get_processed_data_dir, get_models_retrieval_dir
    
    processed_dir = get_processed_data_dir()
    models_dir = get_models_retrieval_dir()
    
    build_hyperbolic_index(
        encoder_path=models_dir / "hyperbolic_encoder",
        base_model_path=models_dir / "zalo_v1",
        chunks_path=processed_dir / "zalo-legal" / "chunks_schema.jsonl",
        output_dir=models_dir / "index_hyperbolic"
    )


if __name__ == "__main__":
    main()
```

---

### 3.4 Phase 4: Fusion Retrieval

#### File: `src/retrieval/hyperbolic_service.py`

```python
"""
HyperbolicRetrievalService: Dual-space retrieval với Mutual-Ranking Fusion.
"""

import json
import numpy as np
import faiss
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from sentence_transformers import SentenceTransformer


class HyperbolicRetrievalService:
    """
    Retrieval service kết hợp Euclidean + Hyperbolic spaces.
    
    Sử dụng Mutual-Ranking Fusion để kết hợp kết quả từ 2 spaces.
    """
    
    def __init__(
        self,
        euclidean_index_dir: Path,
        hyperbolic_index_dir: Path,
        encoder_dir: Path,
        use_gpu: bool = False
    ):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Load Euclidean components (existing)
        self._load_euclidean_index(euclidean_index_dir)
        
        # Load Hyperbolic components (new)
        self._load_hyperbolic_index(hyperbolic_index_dir)
        self._load_hyperbolic_encoder(encoder_dir)
    
    def _load_euclidean_index(self, index_dir: Path):
        """Load existing Euclidean FAISS index."""
        index_path = index_dir / "chunks_index.faiss"
        self.euclidean_index = faiss.read_index(str(index_path))
        
        # Load id_map
        self.euclidean_id_map = {}
        id_map_path = index_dir / "id_map.jsonl"
        with open(id_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.euclidean_id_map[entry['faiss_id']] = entry
        
        # Load base encoder
        model_info_path = index_dir / "model_info.json"
        with open(model_info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        model_path = model_info.get('model_path')
        self.base_encoder = SentenceTransformer(model_path, device="cpu")
        if self.device == "cuda":
            self.base_encoder.to(self.device)
    
    def _load_hyperbolic_index(self, index_dir: Path):
        """Load Hyperbolic FAISS index."""
        index_path = index_dir / "hyperbolic_index.faiss"
        self.hyperbolic_index = faiss.read_index(str(index_path))
        
        # Load id_map
        self.hyperbolic_id_map = {}
        id_map_path = index_dir / "hyperbolic_id_map.jsonl"
        with open(id_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.hyperbolic_id_map[entry['faiss_id']] = entry
    
    def _load_hyperbolic_encoder(self, encoder_dir: Path):
        """Load Poincaré projection layer."""
        checkpoint = torch.load(
            encoder_dir / "hyperbolic_encoder.pt",
            map_location="cpu"
        )
        
        from scripts.train_hyperbolic import PoincareProjection
        
        self.poincare_proj = PoincareProjection(
            input_dim=self.base_encoder.get_sentence_embedding_dimension(),
            output_dim=checkpoint['hyperbolic_dim'],
            curvature=checkpoint['curvature']
        )
        self.poincare_proj.load_state_dict(checkpoint['poincare_proj'])
        self.poincare_proj.to(self.device)
        self.poincare_proj.eval()
    
    def encode_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode query thành cả Euclidean và Hyperbolic embeddings.
        
        Returns:
            (euclidean_emb, hyperbolic_emb)
        """
        # Euclidean
        euclidean = self.base_encoder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)
        faiss.normalize_L2(euclidean)
        
        # Hyperbolic
        with torch.no_grad():
            euclidean_tensor = torch.tensor(euclidean, device=self.device)
            hyperbolic = self.poincare_proj(euclidean_tensor)
            hyperbolic = hyperbolic.cpu().numpy().astype(np.float32)
        
        return euclidean, hyperbolic
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "mutual_ranking",
        euclidean_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents sử dụng dual-space search với fusion.
        
        Args:
            query: Query string
            top_k: Number of results to return
            fusion_method: "mutual_ranking" hoặc "weighted_sum"
            euclidean_weight: Weight cho Euclidean scores (0-1)
        
        Returns:
            List of retrieved documents với scores
        """
        # Encode query
        q_euclidean, q_hyperbolic = self.encode_query(query)
        
        # Search cả 2 indexes (lấy nhiều hơn để fusion)
        k_oversample = top_k * 3
        
        # Euclidean search
        e_distances, e_indices = self.euclidean_index.search(q_euclidean, k_oversample)
        
        # Hyperbolic search
        h_distances, h_indices = self.hyperbolic_index.search(q_hyperbolic, k_oversample)
        
        # Fusion
        if fusion_method == "mutual_ranking":
            results = self._mutual_ranking_fusion(
                e_indices[0], e_distances[0],
                h_indices[0], h_distances[0],
                top_k
            )
        else:
            results = self._weighted_sum_fusion(
                e_indices[0], e_distances[0],
                h_indices[0], h_distances[0],
                top_k,
                euclidean_weight
            )
        
        return results
    
    def _mutual_ranking_fusion(
        self,
        e_indices: np.ndarray,
        e_distances: np.ndarray,
        h_indices: np.ndarray,
        h_distances: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Mutual Ranking Fusion: ưu tiên documents xuất hiện trong cả 2 results.
        
        Documents trong cả Euclidean và Hyperbolic results được boost.
        """
        # Map chunk_id -> scores
        scores = {}
        
        # Euclidean results
        for rank, (idx, dist) in enumerate(zip(e_indices, e_distances)):
            if idx < 0:
                continue
            entry = self.euclidean_id_map.get(int(idx), {})
            chunk_id = entry.get('chunk_id')
            if chunk_id is None:
                continue
            
            # RRF score
            rrf_score = 1.0 / (rank + 60)  # k=60 constant
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    'chunk_id': chunk_id,
                    'corpus_id': entry.get('corpus_id'),
                    'euclidean_rank': rank,
                    'euclidean_score': float(dist),
                    'hyperbolic_rank': None,
                    'hyperbolic_score': None,
                    'rrf_score': rrf_score,
                    'in_both': False
                }
            else:
                scores[chunk_id]['euclidean_rank'] = rank
                scores[chunk_id]['euclidean_score'] = float(dist)
                scores[chunk_id]['rrf_score'] += rrf_score
        
        # Hyperbolic results
        for rank, (idx, dist) in enumerate(zip(h_indices, h_distances)):
            if idx < 0:
                continue
            entry = self.hyperbolic_id_map.get(int(idx), {})
            chunk_id = entry.get('chunk_id')
            if chunk_id is None:
                continue
            
            rrf_score = 1.0 / (rank + 60)
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    'chunk_id': chunk_id,
                    'corpus_id': entry.get('corpus_id'),
                    'euclidean_rank': None,
                    'euclidean_score': None,
                    'hyperbolic_rank': rank,
                    'hyperbolic_score': float(dist),
                    'rrf_score': rrf_score,
                    'in_both': False
                }
            else:
                scores[chunk_id]['hyperbolic_rank'] = rank
                scores[chunk_id]['hyperbolic_score'] = float(dist)
                scores[chunk_id]['rrf_score'] += rrf_score
                scores[chunk_id]['in_both'] = True
        
        # Boost documents in both results
        for chunk_id, info in scores.items():
            if info['in_both']:
                info['rrf_score'] *= 1.5  # 50% boost
        
        # Sort by RRF score
        results = sorted(scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        return results[:top_k]
    
    def _weighted_sum_fusion(
        self,
        e_indices: np.ndarray,
        e_distances: np.ndarray,
        h_indices: np.ndarray,
        h_distances: np.ndarray,
        top_k: int,
        euclidean_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Weighted sum của normalized scores từ 2 spaces.
        """
        hyperbolic_weight = 1.0 - euclidean_weight
        
        scores = {}
        
        # Normalize distances to [0, 1] (lower is better -> invert)
        e_max = e_distances.max() + 1e-6
        h_max = h_distances.max() + 1e-6
        
        # Euclidean
        for idx, dist in zip(e_indices, e_distances):
            if idx < 0:
                continue
            entry = self.euclidean_id_map.get(int(idx), {})
            chunk_id = entry.get('chunk_id')
            if chunk_id is None:
                continue
            
            normalized_score = 1.0 - (dist / e_max)
            weighted_score = normalized_score * euclidean_weight
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    'chunk_id': chunk_id,
                    'corpus_id': entry.get('corpus_id'),
                    'weighted_score': weighted_score
                }
            else:
                scores[chunk_id]['weighted_score'] += weighted_score
        
        # Hyperbolic
        for idx, dist in zip(h_indices, h_distances):
            if idx < 0:
                continue
            entry = self.hyperbolic_id_map.get(int(idx), {})
            chunk_id = entry.get('chunk_id')
            if chunk_id is None:
                continue
            
            normalized_score = 1.0 - (dist / h_max)
            weighted_score = normalized_score * hyperbolic_weight
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    'chunk_id': chunk_id,
                    'corpus_id': entry.get('corpus_id'),
                    'weighted_score': weighted_score
                }
            else:
                scores[chunk_id]['weighted_score'] += weighted_score
        
        results = sorted(scores.values(), key=lambda x: x['weighted_score'], reverse=True)
        return results[:top_k]
```

---

## 4. Chạy Pipeline

### Thứ tự thực thi

```bash
# 1. Build hierarchy
python scripts/build_hierarchy.py

# 2. Train hyperbolic encoder
python scripts/train_hyperbolic.py

# 3. Build hyperbolic index
python scripts/build_hyperbolic_index.py

# 4. Test retrieval
python -c "
from src.retrieval.hyperbolic_service import HyperbolicRetrievalService
from src.utils.paths import get_models_retrieval_dir

models_dir = get_models_retrieval_dir()
service = HyperbolicRetrievalService(
    euclidean_index_dir=models_dir / 'index_v2',
    hyperbolic_index_dir=models_dir / 'index_hyperbolic',
    encoder_dir=models_dir / 'hyperbolic_encoder'
)

results = service.retrieve('Quy định về thừa kế theo pháp luật', top_k=5)
for r in results:
    print(r)
"
```

---

## 5. Tích hợp với GeminiRAG

Sau khi hoàn thành Phase 4, cần update `src/rag/gemini_rag.py`:

```python
# Trong GeminiRAG.__init__()
from ..retrieval.hyperbolic_service import HyperbolicRetrievalService

# Thay thế RetrievalService bằng HyperbolicRetrievalService
self.retriever = HyperbolicRetrievalService(
    euclidean_index_dir=models_dir / 'index_v2',
    hyperbolic_index_dir=models_dir / 'index_hyperbolic',
    encoder_dir=models_dir / 'hyperbolic_encoder',
    use_gpu=self.use_gpu
)
```

---

## 6. Timeline Tổng hợp

| Phase | Công việc | Thời gian | Dependencies |
|-------|-----------|-----------|--------------|
| 1 | Build hierarchy | 1-2 ngày | - |
| 2 | Train hyperbolic | 2-3 ngày | Phase 1 |
| 3 | Build index | 1 ngày | Phase 2 |
| 4 | Implement fusion | 1-2 ngày | Phase 3 |
| 5 | Integration & Test | 1 ngày | Phase 4 |

**Tổng thời gian ước tính: 6-9 ngày**

---

## 7. Checklist Hoàn thành

- [ ] Phase 1: Hierarchy data exported
- [ ] Phase 2: Hyperbolic encoder trained
- [ ] Phase 3: Hyperbolic index built
- [ ] Phase 4: Fusion retrieval working
- [ ] Phase 5: Integrated với GeminiRAG
- [ ] Benchmark: So sánh accuracy với baseline
- [ ] Documentation updated
