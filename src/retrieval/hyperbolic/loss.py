"""
Hierarchical Contrastive Loss (Margin-Based)

Enforces parent-child containment relationships in Hyperbolic geometry.
"""

import torch
import torch.nn as nn
import geoopt

class HierarchicalContrastiveLoss(nn.Module):
    """
    Implements passage-to-fact (document-to-article in Legal perspective) alignment.
    Parents (Documents) act as containers for Children (Articles).
    """
    def __init__(self, curvature: float = 1.0, margin: float = 0.1):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=curvature)
        self.margin = margin
        
    def poincare_dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Calculate geodesic distance in Poincare Ball."""
        return self.ball.dist(u, v)

    def forward(
        self, 
        parent_emb: torch.Tensor, 
        pos_child_emb: torch.Tensor, 
        neg_child_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Passage-to-Fact alignment (Parent-to-Child).
        Parent should be closer to its positive child than to a negative child by a margin.
        
        Args:
            parent_emb: [batch, dim] - Hyperbolic embedding of Document
            pos_child_emb: [batch, dim] - Hyperbolic embedding of True Article
            neg_child_emb: [batch, dim] - Hyperbolic embedding of Random Article
            
        Returns:
            loss: Scalar contrastive loss
        """
        # Distance to positive child
        d_pos = self.poincare_dist(parent_emb, pos_child_emb)
        
        # Distance to negative child
        d_neg = self.poincare_dist(parent_emb, neg_child_emb)
        
        # Margin loss: relu(d_pos - d_neg + margin)
        # We want d_pos < d_neg - margin
        loss_p2f = torch.nn.functional.relu(d_pos - d_neg + self.margin)
        
        # Reverse Fact-to-Passage alignment is symmetrical
        # Since we use parent-child pairs uniformly, we can also add f2p
        # d_pos_rev = self.poincare_dist(pos_child_emb, parent_emb) # Dist is symmetric
        # If we had a negative parent: d_neg_parent = self.poincare_dist(pos_child_emb, neg_parent)
        # For simplicity, we stick to Parent-to-Child
        
        return loss_p2f.mean()

