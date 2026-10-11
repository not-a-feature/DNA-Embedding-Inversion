"""Corrector inference utilities for iterative DNA sequence refinement."""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, BertModel

from src.utils import normalize_embeddings


@torch.no_grad()
def reconstruct_iterative(
    corrector_model: nn.Module,
    base_model: nn.Module | None,
    target_embeddings: np.ndarray,
    fm_checkpoint: str,
    tokenizer: Any,
    device: torch.device,
    iterations: int = 5,
    initial_hypotheses: List[str] | None = None,
    batch_size: int = 32,
    normalization_stats: Dict[str, float] | None = None,
    normalization_method: str = "standard",
) -> List[str]:
    """Perform iterative refinement using the Corrector model.

    Parameters
    ----------
    corrector_model : nn.Module
        Trained CorrectorReconstructor.
    base_model : nn.Module | None
        Trained base model for initial hypotheses. Can be None if initial_hypotheses provided.
    target_embeddings : np.ndarray
        Target mean embeddings to invert (N, input_dim). Expected RAW (un-normalized).
    fm_checkpoint : str
        Path/Name of the Foundation Model for re-embedding.
    tokenizer : Any
        Tokenizer for decoding/encoding sequences.
    device : torch.device
        Device for computation.
    iterations : int
        Number of refinement iterations.
    initial_hypotheses : List[str] | None
        Optional initial guesses. If None, generated using base_model.
    batch_size : int
        Batch size.
    normalization_stats : Dict[str, float] | None
        Normalization stats for embeddings.
    normalization_method : str
        Normalization method ('standard' or 'minmax').

    Returns
    -------
    List[str]
        Refined sequences after T iterations.
    """
    corrector_model.eval()
    if base_model:
        base_model.eval()

    num_samples = len(target_embeddings)
    logger = logging.getLogger(__name__)

    # 1. Generate Initial Hypotheses
    if initial_hypotheses is None:
        assert base_model is not None, "Must provide base_model or initial_hypotheses"

        targets_norm = target_embeddings.copy()
        if normalization_stats:
            targets_norm = normalize_embeddings(
                targets_norm, normalization_stats, normalization_method
            )

        initial_hypotheses = []
        for i in range(0, num_samples, batch_size):
            batch_emb = torch.tensor(targets_norm[i : i + batch_size], dtype=torch.float32).to(
                device
            )
            logits = base_model(batch_emb)
            assert logits.ndim == 3, f"Base model output has unexpected shape: {logits.shape}"
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            if hasattr(tokenizer, "decode_batch"):
                batch_seqs = tokenizer.decode_batch(preds)
            else:
                batch_seqs = [tokenizer.decode(p) for p in preds]
            initial_hypotheses.extend(batch_seqs)

    current_hypotheses = initial_hypotheses

    # Normalize targets once for use in corrector loop
    targets_norm = target_embeddings.copy()
    if normalization_stats:
        targets_norm = normalize_embeddings(targets_norm, normalization_stats, normalization_method)
    targets_tensor = torch.tensor(targets_norm, dtype=torch.float32)

    # Load Foundation Model once for re-embedding
    fm_tokenizer = AutoTokenizer.from_pretrained(fm_checkpoint, trust_remote_code=True)

    is_ntv2 = "nucleotide-transformer" in fm_checkpoint.lower()
    if is_ntv2:
        fm_model = AutoModelForMaskedLM.from_pretrained(fm_checkpoint, trust_remote_code=True)
    elif "dnabert" in fm_checkpoint.lower():
        fm_model = BertModel.from_pretrained(fm_checkpoint, trust_remote_code=True)
    else:
        fm_model = AutoModel.from_pretrained(fm_checkpoint, trust_remote_code=True)

    fm_model.to(device)
    fm_model.eval()

    def fast_embed(seqs):
        embs = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch = seqs[i : i + batch_size]
                inputs = fm_tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    add_special_tokens=False,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                if is_ntv2:
                    outputs = fm_model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                else:
                    outputs = fm_model(**inputs)
                    hidden = outputs[0]

                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                mean = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                embs.append(mean.cpu().numpy())
        return np.concatenate(embs, axis=0)

    # Iterative Refinement Loop
    for step in range(iterations):
        logger.info(f"Corrector Iteration {step + 1}/{iterations}")

        # 1. Re-embed current hypotheses and normalize
        raw_hyp_embs = fast_embed(current_hypotheses)
        hyp_embs = raw_hyp_embs.copy()
        if normalization_stats:
            hyp_embs = normalize_embeddings(hyp_embs, normalization_stats, normalization_method)
        hyp_embs_tensor = torch.tensor(hyp_embs, dtype=torch.float32)

        # 2. Tokenize current hypotheses
        hyp_ids_list = [tokenizer.encode(h) for h in current_hypotheses]

        max_len = max(len(ids) for ids in hyp_ids_list)
        model_seq_len = corrector_model.seq_length
        # Clamp to model capacity
        max_len = min(max_len, model_seq_len)

        hyp_ids_tensor = torch.zeros((num_samples, max_len), dtype=torch.long)
        for i, ids in enumerate(hyp_ids_list):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            l = min(len(ids), max_len)
            hyp_ids_tensor[i, :l] = torch.tensor(ids[:l])

        # 3. Predict refined sequence
        new_hypotheses = []
        for i in range(0, num_samples, batch_size):
            batch_target = targets_tensor[i : i + batch_size].to(device)
            batch_hyp_emb = hyp_embs_tensor[i : i + batch_size].to(device)
            batch_hyp_ids = hyp_ids_tensor[i : i + batch_size].to(device)

            logits = corrector_model(batch_target, batch_hyp_emb, batch_hyp_ids)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            if hasattr(tokenizer, "decode_batch"):
                batch_seqs = tokenizer.decode_batch(preds)
            else:
                batch_seqs = [tokenizer.decode(p) for p in preds]
            new_hypotheses.extend(batch_seqs)

        # Early stopping if sequences have converged
        if new_hypotheses == current_hypotheses:
            logger.info(f"Converged at iteration {step + 1} (no sequence changes)")
            break

        current_hypotheses = new_hypotheses

    return current_hypotheses
