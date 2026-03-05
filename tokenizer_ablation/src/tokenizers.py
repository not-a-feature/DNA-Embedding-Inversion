"""Tokenizer abstractions for DNA sequence models.

This module provides a unified interface for different tokenization strategies:
- Character-level (A, C, G, T)
- HuggingFace AutoTokenizers (BPE, K-mers, etc.)
"""

from __future__ import annotations

import abc
from typing import List, Union
import numpy as np
import torch
from transformers import AutoTokenizer

from src.utils import NUCLEOTIDES


class BaseTokenizer(abc.ABC):
    """Abstract base class for all tokenizers."""

    @abc.abstractproperty
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        pass

    @abc.abstractmethod
    def encode(self, sequence: str) -> torch.Tensor:
        """Encode a sequence string into a LongTensor of token indices.

        Args:
            sequence: Input DNA sequence string.

        Returns:
            torch.LongTensor of shape (seq_length,)
        """
        pass

    @abc.abstractmethod
    def decode(self, indices: Union[torch.Tensor, np.ndarray, List[int]]) -> str:
        """Decode token indices back into a string.

        Args:
            indices: Tensor, array, or list of token indices.

        Returns:
            Decoded sequence string.
        """
        pass


class CharacterTokenizer(BaseTokenizer):
    """Simple character-level tokenizer for A, C, G, T.

    Maps:
        A -> 0
        C -> 1
        G -> 2
        T -> 3
    """

    def __init__(self):
        self.nucleotide_map = {n: i for i, n in enumerate(NUCLEOTIDES)}
        self.inverse_map = {i: n for n, i in self.nucleotide_map.items()}

    @property
    def vocab_size(self) -> int:
        return len(NUCLEOTIDES)

    def encode(self, sequence: str) -> torch.Tensor:
        # Validate characters
        upper_seq = sequence.upper()
        # Fast generic implementation
        indices = [self.nucleotide_map[n] for n in upper_seq]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: Union[torch.Tensor, np.ndarray, List[int]]) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        return "".join(self.inverse_map[i] for i in indices)


class HuggingFaceTokenizer(BaseTokenizer):
    """Wrapper around HuggingFace AutoTokenizer."""

    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, sequence: str) -> torch.Tensor:
        ids = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.squeeze(0)

    def decode(self, indices: Union[torch.Tensor, np.ndarray, List[int]]) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        decoded = self.tokenizer.decode(indices, skip_special_tokens=True)
        return decoded.replace(" ", "")
