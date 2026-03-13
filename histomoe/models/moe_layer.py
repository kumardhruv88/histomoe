"""
MoE Layer — Mixture-of-Experts routing and aggregation core.

Combines gating weights from ``GatingNetwork`` with predictions from
K independent ``ExpertHead`` modules to produce final gene expression output.

Two aggregation modes
---------------------
  - ``'soft'``: Weighted sum of all expert outputs  → output = Σ w_k * expert_k(x)
  - ``'hard'``: Only the top-1 expert is used per token → winner-takes-all

Usage
-----
    from histomoe.models.moe_layer import MoELayer
    layer = MoELayer(
        input_dim=512, output_dim=250,
        num_experts=5, gating_mode="soft"
    )
    embeddings = torch.randn(4, 512)
    labels = torch.tensor([0, 2, 1, 4])
    out, weights, lb_loss = layer(embeddings, labels)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from histomoe.models.expert import ExpertHead
from histomoe.models.gating_network import GatingNetwork


class MoELayer(nn.Module):
    """Core Mixture-of-Experts layer.

    Maintains K independent ``ExpertHead`` modules and a ``GatingNetwork``.
    During a forward pass:
      1. The gating network produces routing weights over K experts.
      2. Each expert independently maps the input to gene predictions.
      3. Expert outputs are combined via weighted summation (soft) or
         winner-takes-all selection (hard).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embedding (fused vision + metadata).
    output_dim : int
        Number of gene expression predictions (= number of target genes).
    num_experts : int
        Number of cancer/organ-specific expert models.
    gating_mode : str
        ``'soft'`` or ``'topk'``.
    top_k : int
        Number of active experts in ``'topk'`` mode.
    expert_hidden_dims : list of int, optional
        Hidden layer sizes for each expert MLP.
    expert_dropout : float
        Dropout within each expert MLP.
    gating_dropout : float
        Dropout within the gating MLP.
    load_balance_weight : float
        Coefficient for load-balancing auxiliary loss.
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 250,
        num_experts: int = 5,
        gating_mode: str = "soft",
        top_k: int = 2,
        expert_hidden_dims: Optional[List[int]] = None,
        expert_dropout: float = 0.1,
        gating_dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim

        # K independent expert heads
        self.experts = nn.ModuleList([
            ExpertHead(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=expert_hidden_dims,
                dropout=expert_dropout,
            )
            for _ in range(num_experts)
        ])

        # Gating network
        self.gating = GatingNetwork(
            input_dim=input_dim,
            num_experts=num_experts,
            mode=gating_mode,
            top_k=top_k,
            dropout=gating_dropout,
            load_balance_weight=load_balance_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        cancer_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route embeddings through experts and aggregate predictions.

        Parameters
        ----------
        x : torch.Tensor
            Fused embedding of shape ``[B, input_dim]``.
        cancer_type_ids : torch.Tensor, optional
            Integer cancer-type labels ``[B]``. Currently unused at forward
            time (routing is fully learned), but retained for potential
            supervised gating ablations.

        Returns
        -------
        output : torch.Tensor
            Aggregated gene expression predictions ``[B, output_dim]``.
        routing_weights : torch.Tensor
            Expert routing weights ``[B, num_experts]``.
        load_balance_loss : torch.Tensor
            Scalar auxiliary load-balancing loss.
        """
        # Step 1: compute routing weights
        routing_weights, lb_loss = self.gating(x)  # [B, K], scalar

        # Step 2: collect all expert outputs → [B, K, G]
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # [B, K, G]

        # Step 3: weighted aggregation
        # routing_weights: [B, K] → [B, K, 1] for broadcasting
        weights_expanded = routing_weights.unsqueeze(-1)       # [B, K, 1]
        output = (expert_outputs * weights_expanded).sum(dim=1)  # [B, G]

        return output, routing_weights, lb_loss

    def get_dominant_expert(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Return the index of the highest-weight expert per sample.

        Parameters
        ----------
        routing_weights : torch.Tensor
            Routing weights ``[B, K]``.

        Returns
        -------
        torch.Tensor
            Expert indices ``[B]`` with the highest weight per sample.
        """
        return routing_weights.argmax(dim=-1)

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        return (
            f"MoELayer(experts={self.num_experts}, "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"total_params={total_params:.1f}M)"
        )
