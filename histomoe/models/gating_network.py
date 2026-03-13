"""
Gating Network — the routing brain of HistoMoE.

Takes a joint vision + metadata embedding and produces routing weights
over K cancer-specific expert models.

Two gating strategies
---------------------
  - ``'soft'`` (default): Softmax over all experts; weighted combination.
    Differentiable, smooth; all experts contribute to every prediction.
  - ``'topk'``: Sparse routing — only the top-K experts are activated.
    Reduces compute; promotes specialisation via load-balancing loss.

Load-Balancing Loss
-------------------
To prevent all tokens routing to a single expert, the gating network
optionally computes an auxiliary load-balancing loss following Switch
Transformer (Fedus et al, 2022):

    L_lb = K * Σ_i f_i * P_i

where ``f_i`` is the fraction of tokens routed to expert i and ``P_i``
is the mean routing probability for expert i.

Usage
-----
    from histomoe.models.gating_network import GatingNetwork
    gate = GatingNetwork(input_dim=512, num_experts=5, mode="soft")
    vision_emb = torch.randn(4, 512)
    weights, lb_loss = gate(vision_emb)  # [4, 5], scalar
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """Learnable routing network that maps embeddings to expert weights.

    Parameters
    ----------
    input_dim : int
        Dimension of the fused (vision + metadata) embedding.
    num_experts : int
        Number of expert models to route to.
    mode : str
        Gating mode: ``'soft'`` (full softmax) or ``'topk'`` (sparse top-K).
    top_k : int
        Number of active experts per token in ``'topk'`` mode.
    hidden_dim : int, optional
        Hidden layer size. Defaults to ``input_dim // 2``.
    dropout : float
        Dropout probability in the gating MLP.
    load_balance_weight : float
        Coefficient for the auxiliary load-balancing loss.
    noise_epsilon : float
        Noise scale for noisy top-K gating (encourages exploration).
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_experts: int = 5,
        mode: str = "soft",
        top_k: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
        noise_epsilon: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.mode = mode
        self.top_k = min(top_k, num_experts)
        self.load_balance_weight = load_balance_weight
        self.noise_epsilon = noise_epsilon

        hidden = hidden_dim or (input_dim // 2)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and load-balancing loss.

        Parameters
        ----------
        x : torch.Tensor
            Fused embedding of shape ``[B, input_dim]``.

        Returns
        -------
        weights : torch.Tensor
            Expert routing weights of shape ``[B, num_experts]``.
            In ``'soft'`` mode: full softmax probabilities summing to 1.
            In ``'topk'`` mode: sparse weights (zeros for non-top experts).
        load_balance_loss : torch.Tensor
            Scalar auxiliary loss promoting uniform expert utilisation.
        """
        logits = self.mlp(x)  # [B, K]

        if self.mode == "soft":
            weights = F.softmax(logits, dim=-1)
        elif self.mode == "topk":
            weights = self._topk_routing(logits)
        else:
            raise ValueError(f"Unknown gating mode: '{self.mode}'")

        lb_loss = self._load_balance_loss(weights)
        return weights, lb_loss

    def _topk_routing(self, logits: torch.Tensor) -> torch.Tensor:
        """Sparse top-K gating with optional noise for exploration."""
        if self.training and self.noise_epsilon > 0:
            noise = torch.randn_like(logits) * self.noise_epsilon
            logits = logits + noise

        top_k_vals, top_k_idx = logits.topk(self.top_k, dim=-1)  # [B, k]
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits.scatter_(-1, top_k_idx, top_k_vals)
        weights = F.softmax(sparse_logits, dim=-1)
        return weights

    def _load_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary load-balancing loss (Switch Transformer style)."""
        # Fraction of tokens routed to each expert
        f = (weights > 0).float().mean(dim=0)   # [K]
        # Mean routing probability per expert
        P = weights.mean(dim=0)                  # [K]
        lb_loss = self.num_experts * (f * P).sum()
        return self.load_balance_weight * lb_loss

    def get_routing_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute per-sample routing entropy (interpretability metric).

        Higher entropy = more uniform routing (less specialisation).

        Parameters
        ----------
        weights : torch.Tensor
            Routing weights ``[B, K]``.

        Returns
        -------
        torch.Tensor
            Shannon entropy per sample, shape ``[B]``.
        """
        eps = 1e-8
        return -(weights * (weights + eps).log()).sum(dim=-1)

    def __repr__(self) -> str:
        return (
            f"GatingNetwork(num_experts={self.num_experts}, "
            f"mode={self.mode}, top_k={self.top_k})"
        )
