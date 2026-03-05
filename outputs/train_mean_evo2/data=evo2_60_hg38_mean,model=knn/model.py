"""KNN Model definition."""

from __future__ import annotations
from typing import List
import torch
from torch import nn
import numpy as np


class KNNReconstructor(nn.Module):
    """K-Nearest Neighbors Reconstructor.

    Uses the training set as a database to find the closest sequences.
    """

    def __init__(self, input_dim: int, output_dim: int = 4, k: int = 1, chunk_size: int = 65536, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.chunk_size = chunk_size

        # Register buffers for persistence
        # Initialize with empty tensors to ensure they exist in state_dict
        self.register_buffer("train_embeddings", torch.empty(0, input_dim))
        self.register_buffer("train_sequences", torch.empty(0))

    def fit(self, dataloader):
        """Fit the KNN model with training data."""
        embeddings_list = []
        sequences_list = []

        # Iterate over dataloader to collect all data
        for emb, seq in dataloader:
            # emb: (batch, embedding_dim)
            # seq: (batch, seq_len) - token indices
            embeddings_list.append(emb)
            sequences_list.append(seq)

        # Concatenate and store as buffers
        if embeddings_list:
            all_embeddings = torch.cat(embeddings_list, dim=0).cpu()
            
            # Handle potential variable sequence lengths by padding everything to the global max length
            max_seq_len = max(s.size(1) for s in sequences_list)
            padded_sequences = []
            for s in sequences_list:
                if s.size(1) < max_seq_len:
                    padding = torch.full((s.size(0), max_seq_len - s.size(1)), -100, dtype=s.dtype, device=s.device)
                    s = torch.cat([s, padding], dim=1)
                padded_sequences.append(s)
            
            all_sequences = torch.cat(padded_sequences, dim=0).cpu()
        else:
            all_embeddings = torch.empty(0, self.input_dim)
            all_sequences = torch.empty(0)

        self.train_embeddings = all_embeddings
        self.train_sequences = all_sequences

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Find nearest neighbors and return reconstructed sequence."""
        # x: (batch, input_dim)
        
        # Ensure we have data
        if self.train_embeddings.numel() == 0:
            raise RuntimeError("KNN model has not been fitted with data.")

        batch_size = x.shape[0]
        device = x.device
        num_train = self.train_embeddings.shape[0]
        
        # Initialize best distances and indices.
        # We start with infinity for distances.
        best_dists = torch.full((batch_size, self.k), float('inf'), device=device)
        best_indices = torch.full((batch_size, self.k), -1, dtype=torch.long, device=device)
        
        # Process training data in chunks to manage GPU memory usage
        for i in range(0, num_train, self.chunk_size):
            end = min(i + self.chunk_size, num_train)
            
            # Move chunk to the same device as input x
            # train_embeddings is typically on CPU
            chunk_embeddings = self.train_embeddings[i:end].to(device)
            
            # Compute Euclidean distance using efficient matrix operations
            # x: (B, D), chunk: (C, D) => dists: (B, C)
            dists = torch.cdist(x, chunk_embeddings, p=2)
            
            # Find top k within this chunk
            # Check if chunk size is smaller than k
            curr_k = min(self.k, end - i)
            chunk_min_dists, chunk_min_indices = torch.topk(dists, k=curr_k, dim=1, largest=False)
            
            # chunk_min_indices are relative to the chunk (0 to C-1)
            # Convert to global indices
            chunk_global_indices = chunk_min_indices + i
            
            # Merge with current best results
            combined_dists = torch.cat([best_dists, chunk_min_dists], dim=1)
            combined_indices = torch.cat([best_indices, chunk_global_indices], dim=1)
            
            # Select the best k from the combined set
            # Keep dimension 1 size at most k
            final_k = min(self.k, combined_dists.shape[1])
            best_dists, best_k_indices = torch.topk(combined_dists, k=final_k, dim=1, largest=False)
            
            # Retrieve the corresponding original indices
            best_indices = torch.gather(combined_indices, 1, best_k_indices)
            
        # Retrieve neighbor sequences using the best indices
        # best_indices is on GPU, train_sequences is on CPU
        indices_cpu = best_indices.cpu()
        neighbor_seqs = self.train_sequences[indices_cpu]  # (batch, k, seq_len)

        # Convert to one-hot: (batch, k, seq_len, output_dim)
        # Handle potential padding indices (-100) by clamping to 0. 
        # Note: Padding is usually not present in the reference database but could be if collate_fn was used.
        neighbor_seqs_clamped = neighbor_seqs.long().clamp(min=0)
        neighbor_onehot = torch.nn.functional.one_hot(
            neighbor_seqs_clamped, num_classes=self.output_dim
        ).float()

        # Move to x device (output device)
        neighbor_onehot = neighbor_onehot.to(device)

        if self.k == 1:
            return neighbor_onehot.squeeze(1)

        # Average (probabilities)
        avg_onehot = torch.mean(neighbor_onehot, dim=1)  # (batch, seq_len, output_dim)
        
        return avg_onehot
