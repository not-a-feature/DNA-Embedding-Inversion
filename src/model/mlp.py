"""Model definitions: simple configurable MLP for DNA sequence reconstruction."""

from __future__ import annotations

from typing import List
import torch
from torch import nn


class SequenceReconstructor(nn.Module):
    """Per-nucleotide MLP for reconstructing DNA nucleotide logits from per-nucleotide embeddings.

    This model processes per-nucleotide embeddings independently and outputs
    logits for each nucleotide (A, C, G, T).

    For each nucleotide in the sequence:
    - Input: embedding vector of shape (input_dim,)
    - Output: logit vector of shape (output_dim,) where output_dim=4 for ACGT

    Parameters
    ----------
    input_dim : int
        The dimension of each per-nucleotide embedding. Must be positive.
    hidden_dims : List[int]
        A list of integers, where each integer is the size of a hidden layer.
        An empty list corresponds to a simple linear model.
    output_dim : int
        The output dimension (number of nucleotide classes). Must be positive.
        Default is 4 for A, C, G, T.
    dropout : float
        The dropout probability to apply after each ReLU activation. Must be
        between 0.0 and 1.0. A value of 0.0 means no dropout.

    Raises
    ------
    AssertionError
        If `input_dim` is not positive, `output_dim` is not positive,
        or `dropout` is outside the [0, 1] range.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert input_dim > 0, "Embedding dimension must be positive."
        assert output_dim > 0, "Output dimension must be positive."
        assert 0.0 <= dropout <= 1.0, "Dropout must be between 0.0 and 1.0."

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input embeddings tensor of shape (batch_size, seq_length, input_dim).

        Returns
        -------
        torch.Tensor
            The reconstructed nucleotide logits of shape (batch_size, seq_length, output_dim).
            Output dimension is 4 for nucleotide classes (A, C, G, T).
        """
        batch_size, seq_length, input_dim = x.shape

        # Reshape to (batch_size * seq_length, input_dim) for MLP
        x_flat = x.reshape(batch_size * seq_length, input_dim)

        # Apply MLP
        output_flat = self.net(x_flat)

        # Reshape back to (batch_size, seq_length, output_dim)
        output_dim = output_flat.shape[1]
        output = output_flat.reshape(batch_size, seq_length, output_dim)

        return output


class SequenceMeanReconstructor(nn.Module):
    """Mean Nucleotide MLP for reconstructing DNA nucleotide logits from per-nucleotide embeddings.

    This model takes the mean and outputs the squence nucleotide (A, C, G, T, N).

    For each nucleotide in the sequence:
    - Input: embedding vector of shape (input_dim,)
    - Output: logit vector of shape (seq_length * one_hot_dim)

    Parameters
    ----------
    input_dim : int
        The dimension of mean embedding. Must be positive.
    hidden_dims : List[int]
        A list of integers, where each integer is the size of a hidden layer.
        An empty list corresponds to a simple linear model.
    seq_length : int
        The length of the DNA sequence to reconstruct. Must be positive.
    output_dim : int
        The dimension of the output at each position (e.g. 4 for ACGT or vocabulary size). Must be positive.
    dropout : float
        The dropout probability to apply after each ReLU activation. Must be
        between 0.0 and 1.0. A value of 0.0 means no dropout.

    Raises
    ------
    AssertionError
        If `input_dim` is not positive, `seq_length` is not positive,
        `output_dim` is not positive, or `dropout` is outside the [0, 1] range.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        seq_length: int,
        output_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert input_dim > 0, "Embedding dimension must be positive."
        assert seq_length > 0, "Sequence length must be positive."
        assert output_dim > 0, "Output dimension must be positive."
        assert 0.0 <= dropout <= 1.0, "Dropout must be between 0.0 and 1.0."

        self.seq_length = seq_length
        self.output_dim = output_dim
        linear_output_dim = seq_length * output_dim

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], linear_output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input embeddings tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            The reconstructed nucleotide logits of shape (batch_size, seq_length, output_dim).
        """
        batch_size, input_dim = x.shape

        x_flat = x.reshape(batch_size, input_dim)
        output_flat = self.net(x_flat)

        output = output_flat.view(batch_size, self.seq_length, self.output_dim)

        return output
