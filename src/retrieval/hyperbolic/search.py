"""
Hyperbolic Search Utilities

Computes exact Poincaré distances using NumPy broadcasting.
This is used as an alternative to FAISS since FAISS does not natively
support true Poincaré curvature distances.
"""

import numpy as np
from typing import Tuple, List

def search_poincare_space(
    query: np.ndarray, 
    corpus_embeddings: np.ndarray,
    top_k: int = 5,
    c: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds top_k closest points in Poincare Ball from corpus to the query.
    
    Formula:
    dist_c(x, y) = arccosh(1 + 2 * c * ||x - y||^2 / ((1 - c * ||x||^2)(1 - c * ||y||^2)))
    
    Args:
        query: [1, dim] vector
        corpus_embeddings: [N, dim] matrix
        top_k: Number of neighbors
        c: Curvature c
        
    Returns:
        (distances, indices): Tuple of Top-K closest embeddings
    """
    if query.ndim == 1:
        query = query.reshape(1, -1)
        
    if corpus_embeddings.ndim == 1:
        corpus_embeddings = corpus_embeddings.reshape(1, -1)
        
    # Broadcasting dimensions
    x = query                        # [1, dim]
    y = corpus_embeddings            # [N, dim]
    
    # 1. Norms squared
    norm_x = np.sum(x**2, axis=-1, keepdims=True)  # [1, 1]
    norm_y = np.sum(y**2, axis=-1)  # [N]
    
    # 2. Difference squared norm
    # ||x - y||^2
    diff = x - y  # [1, N, dim] (broadcast) or [N, dim]
    norm_diff = np.sum(diff**2, axis=-1)  # [N]
    
    # 3. Denominators
    denom = (1 - c * norm_x.flatten()[0]) * (1 - c * norm_y)  # [N]
    denom = np.maximum(denom, 1e-15)  # Numeric stability
    
    # 4. Argument to arccosh
    arg = 1 + 2 * c * norm_diff / denom
    
    # Numeric stability for arccosh (arg >= 1.0)
    arg = np.maximum(arg, 1.0 + 1e-15)
    
    # 5. Distance
    dists = np.arccosh(arg)
    
    if c != 1.0:
        dists = dists * (1.0 / np.sqrt(c))
        
    # Sorting to get top K
    indices = np.argsort(dists)
    top_indices = indices[:top_k]
    top_distances = dists[top_indices]
    
    return top_distances, top_indices

