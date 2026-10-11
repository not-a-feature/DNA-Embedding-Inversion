"""Corrector model for iterative DNA sequence refinement.

Encoder-Decoder architecture inspired by Vec2Text (Morris et al., 2023).
The encoder processes conditioning signals (target embedding + hypothesis embedding)
and the decoder uses cross-attention to refine hypothesis tokens into corrected tokens.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import nn

from src.model.encoder import PositionalEncoding


class CorrectorReconstructor(nn.Module):
    """Encoder-Decoder corrector for iterative refinement of DNA sequences.

    Architecture (Vec2Text-inspired):
      Encoder: Projects target_emb and hypothesis_emb each into N_proj tokens,
               concatenates them into a 2*N_proj conditioning sequence, and
               processes through Transformer encoder layers.
      Decoder: Embeds hypothesis token indices, adds positional encoding,
               and uses Transformer decoder layers with cross-attention to
               the encoder output. Output projection to vocab logits.

    Parameters
    ----------
    input_dim : int
        Dimension of the Foundation Model embeddings.
    seq_length : int
        Length of the DNA sequence (number of output tokens).
    output_dim : int
        Vocabulary size (number of token classes).
    d_model : int
        Internal model dimension.
    n_proj : int
        Number of projection tokens per embedding in the encoder.
    nhead : int
        Number of attention heads.
    num_encoder_layers : int
        Number of Transformer encoder layers.
    num_decoder_layers : int
        Number of Transformer decoder layers.
    dim_feedforward : int
        Feedforward dimension in Transformer layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        output_dim: int,
        d_model: int = 256,
        n_proj: int = 8,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        hidden_dims: Optional[list] = None,
        mode: str = "mean",
    ):
        super().__init__()
        assert input_dim > 0
        assert seq_length > 0
        assert output_dim > 0
        assert d_model > 0
        assert n_proj > 0
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.input_dim = input_dim
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_proj = n_proj

        logger = logging.getLogger(__name__)
        logger.info(
            f"CorrectorReconstructor: input_dim={input_dim}, seq_length={seq_length}, "
            f"d_model={d_model}, n_proj={n_proj}, "
            f"enc_layers={num_encoder_layers}, dec_layers={num_decoder_layers}"
        )

        # --- Encoder side: project embeddings into token sequences ---
        # Each embedding (target, hypothesis) is projected to n_proj tokens of d_model
        self.target_proj = nn.Sequential(
            nn.Linear(input_dim, n_proj * d_model),
            nn.GELU(),
        )
        self.hyp_emb_proj = nn.Sequential(
            nn.Linear(input_dim, n_proj * d_model),
            nn.GELU(),
        )

        # Learned type embeddings to distinguish target vs hypothesis in encoder
        self.type_embedding = nn.Embedding(2, d_model)

        # Encoder positional encoding (max length = 2 * n_proj)
        self.encoder_pos = PositionalEncoding(d_model, max_len=2 * n_proj, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- Decoder side: refine hypothesis tokens via cross-attention ---
        self.token_embedding = nn.Embedding(output_dim, d_model)
        self.decoder_pos = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.01)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        target_embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        hypothesis_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Refine the hypothesis sequence.

        Parameters
        ----------
        target_embedding : torch.Tensor
            Target DNA embedding (batch_size, input_dim).
        hypothesis_embedding : torch.Tensor
            Embedding of the current hypothesis (batch_size, input_dim).
        hypothesis_ids : torch.Tensor
            Current hypothesis token indices (batch_size, seq_length).

        Returns
        -------
        torch.Tensor
            Logits for the refined sequence (batch_size, seq_length, output_dim).
        """
        batch_size = target_embedding.size(0)

        # --- Encoder ---
        # Project target embedding to n_proj tokens
        target_tokens = self.target_proj(target_embedding)  # (B, n_proj * d_model)
        target_tokens = target_tokens.view(batch_size, self.n_proj, self.d_model)

        # Project hypothesis embedding to n_proj tokens
        hyp_tokens = self.hyp_emb_proj(hypothesis_embedding)  # (B, n_proj * d_model)
        hyp_tokens = hyp_tokens.view(batch_size, self.n_proj, self.d_model)

        # Add type embeddings (0 = target, 1 = hypothesis)
        target_tokens = target_tokens + self.type_embedding(
            torch.zeros(self.n_proj, dtype=torch.long, device=target_embedding.device)
        )
        hyp_tokens = hyp_tokens + self.type_embedding(
            torch.ones(self.n_proj, dtype=torch.long, device=target_embedding.device)
        )

        # Concatenate: [target_tokens, hyp_tokens] -> (B, 2*n_proj, d_model)
        encoder_input = torch.cat([target_tokens, hyp_tokens], dim=1)
        encoder_input = self.encoder_pos(encoder_input)
        memory = self.encoder(encoder_input)

        # --- Decoder ---
        # Embed hypothesis tokens
        decoder_input = self.token_embedding(hypothesis_ids)  # (B, seq_length, d_model)
        decoder_input = self.decoder_pos(decoder_input)

        # Cross-attend to encoder memory (no causal mask — parallel prediction)
        decoder_output = self.decoder(decoder_input, memory)

        # Project to logits
        logits = self.output_projection(decoder_output)  # (B, seq_length, output_dim)

        return logits
