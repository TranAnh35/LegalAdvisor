"""
Mutual Ranking Fusion for Dual-Space Retrieval
Integrates FAISS Euclidean rankings and Hyperbolic Numpy Metric rankings.
"""

from typing import List, Dict, Any

def mutual_ranking_fusion(
    results_euclidean: List[Dict[str, Any]], 
    results_hyperbolic: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Implements Mutual-Ranking Fusion from HyperbolicRAG paper.
    
    Args:
        results_euclidean: Top hits from Euclidean search
        results_hyperbolic: Top hits from Hyperbolic search
        top_k: Final number of items to return
        
    Returns:
        fused_results: Reranked dictionaries
    """
    
    # 1. Reciprocal Rank Calculation
    #   s_E(p) = 1 / (rank_E(p) + 1)
    #   s_H(p) = 1 / (rank_H(p) + 1)
    
    def calculate_reciprocal_rank(results_list: List[Dict[str, Any]]) -> Dict[int, float]:
        rr_map = {}
        for rank, item in enumerate(results_list):
            chunk_id = item.get("chunk_id", item.get("id"))
            if chunk_id is not None:
                # 1-indexed rank
                rr_map[chunk_id] = 1.0 / (rank + 1 + 1) 
        return rr_map
        
    rr_E = calculate_reciprocal_rank(results_euclidean)
    rr_H = calculate_reciprocal_rank(results_hyperbolic)
    
    # 2. Consistency Bonus
    #   b(p) = 1 / (rank_E(p) + rank_H(p) + 2) if in BOTH
    #        = 0 otherwise
    
    all_chunk_ids = set(rr_E.keys()).union(set(rr_H.keys()))
    
    final_scores = {}
    
    for cid in all_chunk_ids:
        in_E = cid in rr_E
        in_H = cid in rr_H
        
        s_E = rr_E.get(cid, 0.0)
        s_H = rr_H.get(cid, 0.0)
        
        bonus = 0.0
        if in_E and in_H:
            # Reconstruct ranks assuming 1-indexed ranks were rank_E = (1/s_E) - 1
            rank_E = (1.0 / s_E) - 1.0
            rank_H = (1.0 / s_H) - 1.0
            bonus = 1.0 / (rank_E + rank_H + 2.0)
            
        # 3. Hybrid Score Computation
        #   s_hyb(p) = (s_E(p) + s_H(p)) * (1 + b(p))
        s_hyb = (s_E + s_H) * (1.0 + bonus)
        
        final_scores[cid] = s_hyb
        
    # Build Map to easily retrieve original items
    source_items = {}
    for item in results_euclidean + results_hyperbolic:
        cid = item.get("chunk_id", item.get("id"))
        if cid and cid not in source_items:
            source_items[cid] = item
            
    # Sort and return
    ranked_chunks = sorted(final_scores.keys(), key=lambda c: final_scores[c], reverse=True)
    
    fused_results = []
    for cid in ranked_chunks[:top_k]:
        item = source_items[cid].copy()
        item["hyperbolic_mrf_score"] = final_scores[cid]
        fused_results.append(item)
        
    return fused_results
    
