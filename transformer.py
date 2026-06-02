"""
Stage 2 — Transformer Meta-Detector
=====================================
A Transformer encoder that processes sequences of per-sample forensic
fingerprints across training epochs and produces an anomaly score for
each sample.

Design:
  • Input  : fingerprint sequences  (B, T, D)  — T epochs, D feature dim
  • Encoder: multi-head self-attention over the time axis
  • Output : anomaly score per sample  (B,)  via CLS token projection
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding added to the epoch-time axis.

    Args:
        d_model  : Embedding dimension.
        max_len  : Maximum sequence length (number of epochs).
        dropout  : Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerMetaDetector(nn.Module):
    """
    Transformer encoder that maps per-sample epoch-wise fingerprint sequences
    to a scalar anomaly score.

    Args:
        input_dim    : Dimension of the forensic fingerprint (from Stage 1).
        d_model      : Internal transformer embedding dimension.
        nhead        : Number of attention heads.
        num_layers   : Number of transformer encoder layers.
        dim_ff       : Feed-forward hidden dimension inside the encoder.
        max_epochs   : Maximum number of training epochs (for positional encoding).
        dropout      : Dropout rate.

    Forward:
        x            : Tensor of shape (B, T, input_dim)
                       B = batch samples, T = number of epochs observed
        Returns      : anomaly_scores of shape (B,)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        max_epochs: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project forensic fingerprint → d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable CLS token prepended to each sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(d_model, max_len=max_epochs + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,          # (B, T, D) convention
            norm_first=True,           # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS → scalar anomaly score
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x                   : (B, T, input_dim)  fingerprint sequences
            src_key_padding_mask: (B, T+1) bool mask (True = padding positions)
                                  Pass None if all epochs are valid.

        Returns:
            anomaly_scores : (B,)  higher value = more anomalous
        """
        B, T, _ = x.shape

        # Project and prepend CLS token
        x = self.input_proj(x)                             # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                     # (B, T+1, d_model)

        x = self.pos_enc(x)                                # add positional encoding

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Extract CLS token and project to anomaly score
        cls_out = x[:, 0, :]                               # (B, d_model)
        scores = self.score_head(cls_out).squeeze(-1)      # (B,)
        return scores
