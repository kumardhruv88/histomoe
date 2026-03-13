"""
Metadata / Text Encoder for tissue and cancer-type context.

Encodes the sample-level metadata string (e.g. "Cancer type: CCRCC. Tissue: kidney.")
into an embedding vector for the gating network.

Two encoding strategies
-----------------------
  1. **Lookup** (default, no external deps): a learned embedding table over
     the small cancer-type vocabulary — fast and lightweight.
  2. **BERT** (optional): a frozen ``distilbert-base-uncased`` encoder whose
     [CLS] token is projected to ``embed_dim``.

Usage
-----
    from histomoe.models.text_encoder import MetadataEncoder
    enc = MetadataEncoder(mode="lookup", vocab_size=5, embed_dim=256)
    labels = torch.tensor([0, 2, 1])   # cancer-type indices
    out = enc(labels)                  # [3, 256]
"""

from typing import List, Optional

import torch
import torch.nn as nn

from histomoe.data.metadata_utils import num_cancer_types


class MetadataEncoder(nn.Module):
    """Encode tissue/cancer-type metadata into a dense embedding.

    Supports two modes:
      - ``'lookup'``: fast integer-indexed embedding table (few parameters,
        no tokenizer dependency). Receives **integer labels** as input.
      - ``'bert'``: frozen DistilBERT [CLS] embedding projected to
        ``embed_dim``. Receives **string lists** as input.

    Parameters
    ----------
    mode : str
        ``'lookup'`` or ``'bert'``.
    vocab_size : int
        Number of cancer types (= number of rows in embedding table).
        Only used in ``'lookup'`` mode.
    embed_dim : int
        Output embedding dimension.
    dropout : float
        Dropout applied after the projection.
    bert_model_name : str
        HuggingFace model name for BERT mode.
    """

    def __init__(
        self,
        mode: str = "lookup",
        vocab_size: Optional[int] = None,
        embed_dim: int = 256,
        dropout: float = 0.1,
        bert_model_name: str = "distilbert-base-uncased",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self._vocab_size = vocab_size or num_cancer_types()

        if mode == "lookup":
            self._build_lookup(dropout)
        elif mode == "bert":
            self._build_bert(bert_model_name, dropout)
        else:
            raise ValueError(f"Unknown metadata encoder mode: '{mode}'. Choose 'lookup' or 'bert'.")

    # ------------------------------------------------------------------
    # Mode builders
    # ------------------------------------------------------------------

    def _build_lookup(self, dropout: float) -> None:
        """Build a simple embedding lookup table."""
        self.embedding = nn.Embedding(self._vocab_size, self.embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
        )

    def _build_bert(self, model_name: str, dropout: float) -> None:
        """Load a frozen DistilBERT backbone and projection head."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers is needed for BERT mode: pip install transformers"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._bert = AutoModel.from_pretrained(model_name)
        for p in self._bert.parameters():
            p.requires_grad = False  # frozen

        bert_dim = self._bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        metadata_strings: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Encode cancer-type metadata into embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Integer cancer-type indices of shape ``[B]``.
            Used directly in ``'lookup'`` mode.
        metadata_strings : list of str, optional
            Raw metadata strings of length ``B``.
            Required in ``'bert'`` mode; ignored in ``'lookup'`` mode.

        Returns
        -------
        torch.Tensor
            Metadata embeddings of shape ``[B, embed_dim]``.
        """
        if self.mode == "lookup":
            emb = self.embedding(x)       # [B, embed_dim]
            return self.proj(emb)         # [B, embed_dim]

        elif self.mode == "bert":
            if metadata_strings is None:
                raise ValueError("metadata_strings is required in BERT mode.")
            device = next(self.parameters()).device
            enc = self.tokenizer(
                metadata_strings,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = self._bert(**enc)
            cls_emb = out.last_hidden_state[:, 0, :]  # [B, bert_dim]
            return self.proj(cls_emb)                  # [B, embed_dim]

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        return (
            f"MetadataEncoder(mode={self.mode}, "
            f"embed_dim={self.embed_dim}, trainable_params={n_params:.2f}M)"
        )
