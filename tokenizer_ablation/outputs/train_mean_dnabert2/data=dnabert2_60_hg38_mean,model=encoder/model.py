"""Encoder models for DNA sequence reconstruction from embeddings."""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for encoder.

    Adds position information to the sequence tokens using sine and cosine
    functions of different frequencies.

    Parameters
    ----------
    d_model : int
        The dimension of the model (embedding size).
    max_len : int
        Maximum sequence length to support.
    dropout : float
        Dropout probability to apply after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, d_model).

        Returns
        -------
        torch.Tensor
            Tensor with positional encoding added, same shape as input.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EncoderReconstructor(nn.Module):
    """Unified encoder-based DNA sequence reconstructor for both mean and per-nucleotide embeddings.

    This model handles two modes:
    - per_token: Takes per-nucleotide embeddings (batch_size, seq_length, input_dim)
    - mean: Takes mean embeddings (batch_size, input_dim) and expands to sequence

    Architecture:
    1. Input projection (mode-dependent: expansion for mean, direct projection for per_token)
    2. Positional encoding
    3. Transformer encoder blocks
    4. Output projection to nucleotide logits

    Parameters
    ----------
    input_dim : int
        The dimension of embeddings. Must be positive.
    mode : str
        Either "per_token" or "mean".
    seq_length : int
        The length of the DNA sequence to reconstruct. Must be positive.
    d_model : int
        The dimension of the model. Must be positive.
    nhead : int
        Number of attention heads. Must divide d_model evenly.
    num_layers : int
        Number of encoder layers. Must be positive.
    dim_feedforward : int
        Dimension of the feedforward network. Must be positive.
    dropout : float
        Dropout probability. Must be between 0.0 and 1.0.
    output_dim : int
        Number of nucleotide classes or tokens. Must be positive.
    hidden_dims : Optional[list]
        For compatibility with config. Not used.

    Raises
    ------
    AssertionError
        If any dimension is not positive, dropout is outside [0,1],
        nhead does not divide d_model evenly, or mode is invalid.
    """

    def __init__(
        self,
        input_dim: int,
        mode: str,
        seq_length: int,
        output_dim: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        hidden_dims: Optional[list] = None,  # For compatibility with config
    ):
        super().__init__()
        assert input_dim > 0, "Input dimension must be positive"
        assert mode in [
            "per_token",
            "mean",
        ], f"Invalid mode: {mode}. Must be 'per_token' or 'mean'"
        assert seq_length > 0, "Sequence length must be positive"
        assert output_dim > 0, "Output dimension must be positive"
        assert d_model > 0, "Model dimension must be positive"
        assert nhead > 0, "Number of heads must be positive"
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        assert num_layers > 0, "Number of layers must be positive"
        assert dim_feedforward > 0, "Feedforward dimension must be positive"
        assert 0.0 <= dropout <= 1.0, "Dropout must be between 0.0 and 1.0"

        self.input_dim = input_dim
        self.mode = mode
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.d_model = d_model

        logger = logging.getLogger(__name__)

        # Mode-dependent input projection
        if mode == "mean":
            # Expand mean embedding to sequence of d_model-dimensional tokens
            self.input_projection = nn.Linear(input_dim, seq_length * d_model)
            logger.info(
                f"Encoder [{mode}]: input_dim={input_dim} -> "
                f"seq_length={seq_length} x d_model={d_model} -> "
                f"output_dim={output_dim}"
            )
        else:  # per_token
            # Project per-nucleotide embeddings to d_model dimension
            self.input_projection = nn.Linear(input_dim, d_model)
            logger.info(
                f"Encoder [{mode}]: input_dim={input_dim} -> "
                f"d_model={d_model} -> "
                f"output_dim={output_dim} (seq_length={seq_length})"
            )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to nucleotide logits
        self.output_projection = nn.Linear(d_model, output_dim)

        # Initialize output projection with smaller weights to prevent large initial outputs
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct DNA sequence from embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings. Shape depends on mode:
            - per_token: (batch_size, seq_length, input_dim)
            - mean: (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Reconstructed nucleotide logits of shape (batch_size, seq_length, output_dim).
        """
        if self.mode == "mean":
            assert x.ndim == 2, f"Expected 2D input (batch, input_dim) for mean mode, got {x.ndim}D"
            batch_size, input_dim = x.shape
            assert (
                input_dim == self.input_dim
            ), f"Expected input_dim={self.input_dim}, got {input_dim}"

            # Project to sequence of tokens: (batch_size, input_dim) -> (batch_size, seq_length * d_model)
            x = self.input_projection(x)

            # Reshape to sequence: (batch_size, seq_length * d_model) -> (batch_size, seq_length, d_model)
            x = x.view(batch_size, self.seq_length, self.d_model)

        else:  # per_token
            assert (
                x.ndim == 3
            ), f"Expected 3D input (batch, seq_length, input_dim) for per_token mode, got {x.ndim}D"
            batch_size, seq_length, input_dim = x.shape
            assert (
                input_dim == self.input_dim
            ), f"Expected input_dim={self.input_dim}, got {input_dim}"
            assert (
                seq_length == self.seq_length
            ), f"Expected seq_length={self.seq_length}, got {seq_length}"

            # Project embeddings to d_model dimension: (batch_size, seq_length, input_dim) -> (batch_size, seq_length, d_model)
            x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Project to nucleotide logits
        x = self.output_projection(x)

        return x
