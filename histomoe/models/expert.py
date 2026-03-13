"""
Expert Head — a single cancer-type specialist prediction module.

Each expert is a lightweight MLP (optionally with residual connections)
that maps vision embeddings to patch-level gene expression predictions
for a specific biological context (cancer type / organ).

Usage
-----
    from histomoe.models.expert import ExpertHead
    expert = ExpertHead(input_dim=512, output_dim=250, hidden_dims=[1024, 512])
    x = torch.randn(4, 512)
    pred = expert(x)  # [4, 250]
"""

from typing import List, Optional

import torch
import torch.nn as nn


class ExpertHead(nn.Module):
    """Cancer-type-specific MLP prediction head.

    Maps patch embeddings ``[B, D]`` → gene expression predictions ``[B, G]``.
    Supports configurable depth, width, activation, dropout, and optional
    residual connections within the hidden layers.

    Parameters
    ----------
    input_dim : int
        Dimension of the incoming patch embedding.
    output_dim : int
        Number of gene expression targets to predict.
    hidden_dims : list of int
        Width of each hidden layer. Defaults to ``[input_dim * 2, input_dim]``.
    dropout : float
        Dropout probability applied after each hidden layer.
    use_residual : bool
        If True and consecutive hidden dims match, add identity skip connections.
    activation : str
        Activation function: ``'gelu'`` (default), ``'relu'``, or ``'silu'``.
    output_activation : str or None
        Optional activation on the output (e.g., ``'softplus'`` for non-negative
        gene counts). None = linear output.
    """

    _ACTIVATIONS = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 250,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_residual: bool = True,
        activation: str = "gelu",
        output_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

        hidden_dims = hidden_dims or [input_dim * 2, input_dim]
        act_cls = self._ACTIVATIONS.get(activation, nn.GELU)

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], output_dim)

        # Optional residual projections for skip connections
        self._residual_projs = nn.ModuleList()
        if use_residual:
            for i in range(len(dims) - 1):
                if dims[i] != dims[i + 1]:
                    self._residual_projs.append(nn.Linear(dims[i], dims[i + 1], bias=False))
                else:
                    self._residual_projs.append(nn.Identity())

        # Output activation
        if output_activation == "softplus":
            self.out_act: Optional[nn.Module] = nn.Softplus()
        elif output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict gene expression from vision embedding.

        Parameters
        ----------
        x : torch.Tensor
            Patch embedding of shape ``[B, input_dim]``.

        Returns
        -------
        torch.Tensor
            Gene expression predictions of shape ``[B, output_dim]``.
        """
        if self.use_residual and len(self._residual_projs) > 0:
            # Apply hidden layers with residual connections block by block
            # Each block = Linear + LayerNorm + Activation + Dropout (4 submodules)
            h = x
            block_size = 4
            blocks = [self.hidden[i:i+block_size] for i in range(0, len(self.hidden), block_size)]
            for idx, block in enumerate(blocks):
                residual = self._residual_projs[idx](h)
                h_new = h
                for layer in block:
                    h_new = layer(h_new)
                h = h_new + residual
        else:
            h = self.hidden(x)

        out = self.output(h)
        if self.out_act is not None:
            out = self.out_act(out)
        return out

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return (
            f"ExpertHead(input={self.input_dim}, output={self.output_dim}, "
            f"params={n_params:.2f}M)"
        )
