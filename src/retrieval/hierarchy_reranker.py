#!/usr/bin/env python3
"""
Hierarchical Reranker cho LegalAdvisor.

Tận dụng hierarchy data (hierarchy.json, chunk_depths.json) đã xây dựng
để cải thiện chất lượng retrieval thông qua:
1. Query classification (tổng quát vs cụ thể)
2. Depth-aware score adjustment
3. Document-level authority boosting
4. Diversity enforcement (giới hạn articles/document)

Không cần train model mới — chỉ dùng heuristics + hierarchy metadata.

Usage:
    reranker = HierarchyReranker()
    reranked = reranker.rerank(candidates, query="Quy định về thừa kế", top_k=5)

ENV configuration:
    LEGALADVISOR_HIERARCHY_RERANK=1       # 1=enabled (default), 0=disabled
    LEGALADVISOR_HIERARCHY_ALPHA=0.15     # Depth adjustment factor
    LEGALADVISOR_HIERARCHY_MAX_PER_DOC=3  # Max articles per document in results
    LEGALADVISOR_HIERARCHY_AUTHORITY=1    # 1=enable authority boosting
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.paths import get_processed_data_dir


# ============================================================
# Hằng số — Document type authority weights
# ============================================================
# Luật/Bộ luật ban hành bởi Quốc hội có hiệu lực pháp lý cao nhất,
# sau đó là Nghị định (Chính phủ), Thông tư (Bộ), Quyết định, v.v.
AUTHORITY_WEIGHTS: Dict[str, float] = {
    "Quốc hội": 1.10,
    "Chính phủ": 1.05,
    "Thủ tướng": 1.03,
    # Các bộ ban hành thông tư
    "Bộ Tài chính": 1.00,
    "Bộ Y tế": 1.00,
    "Bộ GTVT": 1.00,
    "Bộ Công an": 1.00,
    "Bộ TN&MT": 1.00,
    "Bộ Công Thương": 1.00,
    "Bộ GD&ĐT": 1.00,
    "Bộ LĐ-TB&XH": 1.00,
    "Bộ Tư pháp": 1.00,
    "Bộ TT&TT": 1.00,
    "Bộ NN&PTNT": 1.00,
    "Bộ Xây dựng": 1.00,
}
DEFAULT_AUTHORITY_WEIGHT = 0.98

# ============================================================
# Keywords cho query classification
# ============================================================
# Patterns gợi ý query tổng quát (muốn Điều có tính khái quát cao)
GENERAL_PATTERNS = [
    r"\bquy\s*[đd]ịnh\s+chung\b",
    r"\bnguyên\s*tắc\b",
    r"\b[đd]ịnh\s*nghĩa\b",
    r"\bgiải\s*thích\s*từ\s*ngữ\b",
    r"\bphạm\s*vi\s*[đd]iều\s*chỉnh\b",
    r"\b[đd]ối\s*tượng\s*áp\s*dụng\b",
    r"\bquyền\s+và\s+nghĩa\s*vụ\b",
    r"\bcác\s+loại\b",
    r"\bphân\s*loại\b",
    r"\bkhái\s*niệm\b",
    r"\bhệ\s*thống\b",
    r"\btổng\s*quan\b",
]

# Patterns gợi ý query cụ thể (muốn Điều chi tiết, xử phạt, thủ tục)
SPECIFIC_PATTERNS = [
    r"\bmức\s+phạt\b",
    r"\bxử\s*phạt\b",
    r"\bbao\s+nhiêu\b",
    r"\bthời\s*hạn\b",
    r"\bthủ\s*tục\b",
    r"\bhồ\s*sơ\b",
    r"\b[đd]iều\s*kiện\b",
    r"\btrình\s*tự\b",
    r"\bcụ\s*thể\b",
    r"\btrường\s+hợp\b",
    r"\bvi\s*phạm\b",
    r"\bchế\s*tài\b",
    r"\blệ\s*phí\b",
    r"\bthuế\b",
    r"\bgiá\b",
    r"\bmức\b",
]


class HierarchyReranker:
    """Hierarchy-aware reranker cho retrieval results.

    Load hierarchy metadata từ Phase 1 (build_hierarchy.py) và dùng
    để điều chỉnh ranking dựa trên:
    - Depth của article trong hierarchy (tổng quát vs cụ thể)
    - Authority của cơ quan ban hành (Quốc hội > Chính phủ > Bộ)
    - Diversity giữa các văn bản
    """

    def __init__(self) -> None:
        self._logger = get_logger("legaladvisor.hierarchy_reranker")

        # Đọc cấu hình từ ENV
        self._alpha = self._float_env("LEGALADVISOR_HIERARCHY_ALPHA", 0.15)
        self._max_per_doc = self._int_env("LEGALADVISOR_HIERARCHY_MAX_PER_DOC", 3)
        self._enable_authority = self._bool_env("LEGALADVISOR_HIERARCHY_AUTHORITY", True)

        # Load hierarchy data
        self._chunk_depths: Dict[str, float] = {}
        self._doc_issuers: Dict[str, str] = {}
        self._doc_types: Dict[str, str] = {}
        self._hierarchy_loaded: bool = False

        self._load_hierarchy_data()

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank danh sách candidates dựa trên hierarchy signals.

        Args:
            candidates: Danh sách kết quả từ FAISS retrieval (đã có 'score').
            query: Query string gốc.
            top_k: Số lượng kết quả cần trả về.

        Returns:
            Danh sách đã rerank, mỗi item thêm fields:
            - hierarchy_score: Điểm sau hierarchy adjustment
            - query_type: general/specific/neutral
            - depth: Depth trong hierarchy
            - authority_weight: Hệ số authority
        """
        if not candidates:
            return []

        if not self._hierarchy_loaded:
            # Nếu không load được hierarchy → trả nguyên
            self._logger.debug("Hierarchy data not available, returning original ranking")
            return candidates[:top_k]

        # 1. Phân loại query
        query_type = self.classify_query(query)

        # 2. Tính adjusted score cho từng candidate
        scored = []
        for item in candidates:
            adjusted = self._compute_hierarchy_score(item, query_type)
            scored.append(adjusted)

        # 3. Sắp xếp theo hierarchy_score giảm dần
        scored.sort(key=lambda x: x.get("hierarchy_score", 0.0), reverse=True)

        # 4. Áp dụng diversity enforcement
        diversified = self._enforce_diversity(scored, top_k)

        return diversified

    def classify_query(self, query: str) -> str:
        """Phân loại query thành general / specific / neutral.

        Sử dụng keyword matching + heuristics đơn giản.
        """
        if not query:
            return "neutral"

        text = query.lower().strip()

        general_score = 0
        specific_score = 0

        for pattern in GENERAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                general_score += 1

        for pattern in SPECIFIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                specific_score += 1

        # Heuristic phụ: query ngắn thường tổng quát hơn
        if len(text) < 30:
            general_score += 0.5
        elif len(text) > 80:
            specific_score += 0.5

        # Quyết định
        if general_score > specific_score and general_score >= 1:
            return "general"
        elif specific_score > general_score and specific_score >= 1:
            return "specific"
        return "neutral"

    def get_depth(self, corpus_id: str) -> Optional[float]:
        """Lấy depth của một corpus_id từ hierarchy."""
        return self._chunk_depths.get(corpus_id)

    def get_issuer(self, doc_code: str) -> Optional[str]:
        """Lấy issuer (cơ quan ban hành) của document."""
        return self._doc_issuers.get(doc_code)

    # ----------------------------------------------------------------
    # Internal: Loading
    # ----------------------------------------------------------------

    def _load_hierarchy_data(self) -> None:
        """Load hierarchy.json và chunk_depths.json."""
        processed_dir = get_processed_data_dir()
        zalo_dir = processed_dir / "zalo-legal"

        # 1. Load chunk_depths.json (corpus_id → depth)
        depths_path = zalo_dir / "chunk_depths.json"
        if depths_path.exists():
            try:
                with open(depths_path, "r", encoding="utf-8") as f:
                    self._chunk_depths = json.load(f)
                self._logger.info(
                    "Loaded %d chunk depth mappings from %s",
                    len(self._chunk_depths),
                    depths_path,
                )
            except Exception as exc:
                self._logger.warning("Failed to load chunk_depths.json: %s", exc)
        else:
            self._logger.warning("chunk_depths.json not found at %s", depths_path)

        # 2. Load hierarchy.json (full hierarchy with issuer info)
        hierarchy_path = zalo_dir / "hierarchy.json"
        if hierarchy_path.exists():
            try:
                with open(hierarchy_path, "r", encoding="utf-8") as f:
                    hierarchy = json.load(f)

                nodes = hierarchy.get("nodes", {})
                for node_id, node in nodes.items():
                    node_type = node.get("type", "")

                    # Extract issuer info từ document nodes
                    if node_type == "document":
                        # node_id format: "doc:91/2015/qh13"
                        label = node.get("label", "")
                        issuer = node.get("issuer", "")
                        doc_type_val = node.get("doc_type", "")
                        if label:
                            self._doc_issuers[label] = issuer
                            self._doc_types[label] = doc_type_val

                self._logger.info(
                    "Loaded hierarchy: %d documents, %d issuers mapped",
                    len(self._doc_issuers),
                    len(set(self._doc_issuers.values())),
                )
                self._hierarchy_loaded = True
            except Exception as exc:
                self._logger.warning("Failed to load hierarchy.json: %s", exc)
        else:
            self._logger.warning("hierarchy.json not found at %s", hierarchy_path)

        # Coi như loaded nếu có ít nhất chunk_depths
        if self._chunk_depths and not self._hierarchy_loaded:
            self._hierarchy_loaded = True

    # ----------------------------------------------------------------
    # Internal: Score computation
    # ----------------------------------------------------------------

    def _compute_hierarchy_score(
        self, item: Dict[str, Any], query_type: str
    ) -> Dict[str, Any]:
        """Tính hierarchy-adjusted score cho một candidate.

        Trả về bản sao (copy) của item với các field bổ sung.
        """
        result = dict(item)
        base_score = float(item.get("score", 0.0))

        corpus_id = str(item.get("corpus_id") or "").strip()
        doc_code = corpus_id.split("+")[0].strip() if "+" in corpus_id else corpus_id

        # 1. Depth adjustment
        depth = self._chunk_depths.get(corpus_id)
        depth_bonus = 0.0
        if depth is not None:
            depth_bonus = self._depth_bonus(depth, query_type)

        # 2. Authority boosting
        authority_weight = 1.0
        issuer = ""
        if self._enable_authority and doc_code:
            issuer = self._doc_issuers.get(doc_code, "")
            authority_weight = AUTHORITY_WEIGHTS.get(issuer, DEFAULT_AUTHORITY_WEIGHT)

        # 3. Combine
        hierarchy_score = base_score * (1.0 + depth_bonus) * authority_weight

        # 4. Annotate
        result["hierarchy_score"] = hierarchy_score
        result["_original_score"] = base_score
        result["_query_type"] = query_type
        result["_depth"] = depth
        result["_depth_bonus"] = depth_bonus
        result["_authority_weight"] = authority_weight
        result["_issuer"] = issuer
        result["_doc_code"] = doc_code

        return result

    def _depth_bonus(self, depth: float, query_type: str) -> float:
        """Tính depth bonus dựa trên query type.

        Args:
            depth: Depth trong Poincaré ball [0, 1].
                   0 = root (rất tổng quát), 1 = leaf (rất cụ thể)
            query_type: "general", "specific", hoặc "neutral"

        Returns:
            Bonus multiplier (có thể âm hoặc dương).
        """
        alpha = self._alpha

        if query_type == "general":
            # Ưu tiên articles tổng quát (depth thấp → bonus cao)
            # Depth thấp = gần center = tổng quát hơn
            return (1.0 - depth) * alpha

        elif query_type == "specific":
            # Ưu tiên articles cụ thể (depth cao → bonus cao)
            return depth * alpha

        else:  # neutral
            # Không điều chỉnh depth
            return 0.0

    # ----------------------------------------------------------------
    # Internal: Diversity enforcement
    # ----------------------------------------------------------------

    def _enforce_diversity(
        self, candidates: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Đảm bảo kết quả không tập trung quá nhiều vào 1 văn bản.

        Giới hạn tối đa `max_per_doc` articles từ cùng 1 document.
        Nếu vượt quá, đẩy article thừa xuống cuối danh sách.
        """
        max_per_doc = self._max_per_doc
        if max_per_doc <= 0:
            # 0 = không giới hạn
            return candidates[:top_k]

        doc_counts: Dict[str, int] = {}
        selected: List[Dict[str, Any]] = []
        overflow: List[Dict[str, Any]] = []

        for item in candidates:
            doc_code = item.get("_doc_code", "")
            if not doc_code:
                corpus_id = str(item.get("corpus_id") or "").strip()
                doc_code = (
                    corpus_id.split("+")[0].strip() if "+" in corpus_id else corpus_id
                )

            current_count = doc_counts.get(doc_code, 0)

            if current_count < max_per_doc:
                selected.append(item)
                doc_counts[doc_code] = current_count + 1
            else:
                overflow.append(item)

            if len(selected) >= top_k:
                break

        # Nếu chưa đủ top_k, lấy thêm từ overflow
        if len(selected) < top_k:
            remaining = top_k - len(selected)
            selected.extend(overflow[:remaining])

        return selected[:top_k]

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _float_env(key: str, default: float) -> float:
        try:
            return float(os.getenv(key, str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int_env(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool_env(key: str, default: bool) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.strip().lower() in ("1", "true", "yes", "on")
