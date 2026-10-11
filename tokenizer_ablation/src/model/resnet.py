"""ResNet model definitions for DNA sequence reconstruction from embeddings."""

from __future__ import annotations

import logging
from typing import Optional, List

import torch
from torch import nn


class ResNetBlock(nn.Module):
    """1D Residual Block with two convolutional layers.

    Standard ResNet block: x -> Conv1d -> BN -> ReLU -> Conv1d -> BN -> (+x) -> ReLU

    Parameters
    ----------
    channels : int
        Number of input and output channels (matches d_model).
    kernel_size : int
        Kernel size for the 1D convolutions.
    dropout : float
        Dropout probability.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, seq_len).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, channels, seq_len).
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNetReconstructor(nn.Module):
    """ResNet-based DNA sequence reconstructor.

    Similar to the TransformerReconstructor, this model projects input embeddings
    to the full sequence length and then refines them using a stack of 1D
    Residual Blocks.

    Structure:
    1. Input Projection (Mean -> SeqLen*D or PerToken -> D)
    2. ResNet Backbone (Stack of ResNetBlocks)
    3. Output Projection (D -> OutputDim)

    Parameters
    ----------
    input_dim : int
        Dimension of the input embeddings.
    mode : str
        "mean" or "per_token".
    seq_length : int
        Target sequence length.
    output_dim : int
        Number of nucleotide classes.
    d_model : int
        Model channel dimension.
    n_blocks : int
        Number of ResNet blocks.
    kernel_size : int
        Kernel size for convolutions.
    dropout : float
        Dropout probability.
    hidden_dims : Optional[list]
        Ignored, for config compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        mode: str,
        seq_length: int,
        output_dim: int = 4,
        d_model: int = 256,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        assert input_dim > 0, "Input dimension must be positive"
        assert mode in ["per_token", "mean"], "Invalid mode"
        assert seq_length > 0, "Sequence length must be positive"
        assert d_model > 0, "Model dimension must be positive"
        assert n_blocks > 0, "Number of blocks must be positive"

        self.input_dim = input_dim
        self.mode = mode
        self.seq_length = seq_length
        self.d_model = d_model

        logger = logging.getLogger(__name__)

        # 1. Input Projection
        if mode == "mean":
            # Expand mean: (batch, input_dim) -> (batch, seq_len * d_model)
            self.input_projection = nn.Linear(input_dim, seq_length * d_model)
            logger.info(
                f"ResNet [{mode}]: input_dim={input_dim} -> "
                f"seq_length={seq_length} x d_model={d_model}"
            )
        else:
            # per_token: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
            self.input_projection = nn.Linear(input_dim, d_model)
            logger.info(f"ResNet [{mode}]: input_dim={input_dim} -> d_model={d_model}")

        # 2. Backbone
        layers = []
        for _ in range(n_blocks):
            layers.append(ResNetBlock(channels=d_model, kernel_size=kernel_size, dropout=dropout))
        self.backbone = nn.Sequential(*layers)

        # 3. Output Projection
        self.output_projection = nn.Linear(d_model, output_dim)

        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings.
            - mean: (batch, input_dim)
            - per_token: (batch, seq_len, input_dim)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, seq_len, output_dim).
        """
        batch_size = x.shape[0]

        # 1. Project and Reshape
        if self.mode == "mean":
            x = self.input_projection(x)
            # (batch, seq_len * d_model) -> (batch, seq_len, d_model)
            x = x.view(batch_size, self.seq_length, self.d_model)
        else:
            x = self.input_projection(x)

        # 2. ResNet Backbone (requires Channel-First format for Conv1d)
        # (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)

        x = self.backbone(x)

        # (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2)

        # 3. Output Projection
        x = self.output_projection(x)

        return x
