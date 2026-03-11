"""
Hyperbolic Encoder Module

Projects Euclidean embedding into Poincaré Ball with a learnable depth predictor.
Implements the core idea from HyperbolicRAG paper.
"""

import torch
import torch.nn as nn
import geoopt

class DepthPredictor(nn.Module):
    """
    MLP that projects Euclidean features to a scalar specificity depth score in [0, 1].
    Low score (near 0) = General node
    High score (near 1) = Specific node
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output depth d_v in (0, 1)
        return self.net(x)

class HyperbolicEncoder(nn.Module):
    """
    Projects dense Euclidean embeddings to Hyperbolic space (Poincaré Ball)
    with depth alignment as proposed by HyperbolicRAG.
    """
    def __init__(
        self, 
        input_dim: int = 384, 
        curvature: float = 1.0,
        alpha_depth: float = 0.1,
        beta_depth: float = 0.89
    ):
        """
        Args:
            input_dim: Dimension of Euclidean embedding
            curvature: Curvature c of the Poincare ball
            alpha_depth: Base norm in Poincare ball
            beta_depth: Scaling factor for depth. alpha + beta < 1 to keep inside ball.
        """
        super().__init__()
        self.input_dim = input_dim
        self.curvature = curvature
        
        # Ensures max radius = alpha_depth + beta_depth < 1.0 
        self.alpha_depth = alpha_depth
        self.beta_depth = beta_depth
        
        self.ball = geoopt.PoincareBall(c=curvature)
        
        # Projection Matrix to extract hierarchy features
        self.hierarchy_transform = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )
        
        # Depth predictor
        self.depth_predictor = DepthPredictor(input_dim)
        
        # Gating Weights
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, euclidean_emb: torch.Tensor) -> dict:
        """
        Forward pass embedding into the Poincare ball.
        
        Returns:
            Dict containing:
                "hyperbolic_emb": Projected coordinates in Poincare ball
                "depth": Predicted depth scores
        """
        # Feature extraction (Hierarchy features u_v)
        u_v = self.hierarchy_transform(euclidean_emb)
        
        # Depth Prediction
        d_v = self.depth_predictor(u_v)  # Shape [batch, 1]
        
        # Feature Fusion with Gating
        m_v = torch.sigmoid(self.gate(euclidean_emb))
        z_refined = euclidean_emb * m_v + u_v * (1 - m_v)  # Shape [batch, dim]
        
        # Radial Depth Alignment
        # norm_target = alpha + beta * d_v
        norm_target = self.alpha_depth + self.beta_depth * d_v  # Shape [batch, 1]
        
        # Normalize z_refined to have exactly norm_target length
        current_norm = torch.norm(z_refined, p=2, dim=-1, keepdim=True) + 1e-9
        z_aligned = z_refined * (norm_target / current_norm)
        
        # Map to Hyperbolic Space via exponential map at origin
        # Note: geoopt's expmap0 operates on the tangent space at 0
        z_hyperbolic = self.ball.expmap0(z_aligned)
        
        return {
            "hyperbolic_emb": z_hyperbolic,
            "depth": d_v.squeeze(-1),
            "euclidean_refined": z_refined
        }

