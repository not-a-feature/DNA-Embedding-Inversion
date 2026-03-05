"""Training and evaluation utilities for DNA sequence reconstruction.

This module provides a standardized `fit` function to orchestrate the training
and evaluation loop for sequence reconstruction tasks. It uses MSE loss to
reconstruct one-hot encoded DNA sequences from embeddings.
"""

from __future__ import annotations

from typing import Dict, Callable, Any
import time
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    interval: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    tokenizer: Any | None = None,
) -> Dict[str, float]:
    """Perform one full training epoch.

    Sets the model to training mode, iterates over the data loader, computes
    gradients, and updates the model parameters.

    Returns
    -------
    Dict[str, float]
        A dictionary with the average training loss for the epoch.
    """
    model.train()
    losses = []
    for batch_idx, (batch_embedding, batch_sequence) in enumerate(loader):
        batch_embedding = batch_embedding.to(device)
        batch_sequence = batch_sequence.to(device)

        # Forward pass
        predictions = model(batch_embedding)

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            # Model output: (B, L, V), needs (B, V, L) for CrossEntropyLoss
            # Target: (B, L) indices
            predictions = predictions.permute(0, 2, 1)

            # Match lengths by padding/truncating the target (batch_sequence)
            # This is more robust as it ensures every model output position is accounted for
            # while ignoring positions beyond the actual sequence length using ignore_index=-100.
            target_len = batch_sequence.size(1)
            pred_len = predictions.size(2)

            if target_len < pred_len:
                # Pad target sequence with -100 to match model output length
                padding = torch.full(
                    (batch_sequence.size(0), pred_len - target_len),
                    -100,
                    device=device,
                    dtype=torch.long,
                )
                batch_sequence = torch.cat([batch_sequence, padding], dim=1)
            elif target_len > pred_len:
                # This should rarely happen if effective_seq_length is set correctly
                batch_sequence = batch_sequence[:, :pred_len]

        loss = loss_fn(predictions, batch_sequence)

        if (batch_idx + 1) % interval == 0:
            log_obj = {
                "batch_idx": batch_idx + 1,
                "loss": loss.item(),
            }
            if scheduler is not None:
                log_obj["lr"] = scheduler.get_last_lr()[0]
            log_fn(log_obj)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Step scheduler after each batch (required for OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
    return {"loss": float(np.mean(losses))}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    tokenizer: Any | None = None,
) -> Dict[str, float]:
    """Evaluate the model on a given dataset without computing gradients.

    Sets the model to evaluation mode and computes loss over the provided
    data loader.

    Returns
    -------
    Dict[str, float]
        A dictionary with the average validation loss.
    """
    model.eval()
    losses = []
    for batch_embedding, batch_sequence in loader:
        batch_embedding = batch_embedding.to(device)
        batch_sequence = batch_sequence.to(device)
        predictions = model(batch_embedding)
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            predictions = predictions.permute(0, 2, 1)

            target_len = batch_sequence.size(1)
            pred_len = predictions.size(2)

            if target_len < pred_len:
                padding = torch.full(
                    (batch_sequence.size(0), pred_len - target_len),
                    -100,
                    device=device,
                    dtype=torch.long,
                )
                batch_sequence = torch.cat([batch_sequence, padding], dim=1)
            elif target_len > pred_len:
                batch_sequence = batch_sequence[:, :pred_len]

        loss = loss_fn(predictions, batch_sequence)
        losses.append(loss.item())
    return {"loss": float(np.mean(losses))}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    interval: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    early_stopping_enabled: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-6,
    tokenizer: Any | None = None,
) -> Dict[str, float]:
    """Train a model with validation-based early stopping.

    This function implements a complete training loop for sequence reconstruction.
    It monitors validation loss and saves the state of the best-performing model.
    After training is complete, it restores this best state.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    optimizer : torch.optim.Optimizer | None
        The optimizer for model parameters. Can be None if epochs=0.
    loss_fn : nn.Module
        Loss function (e.g., MSELoss).
    epochs : int
        Number of training epochs.
    device : torch.device
        Device to run training on (cuda or cpu).
    log_fn : Callable[[Dict[str, Any]], None]
        A function to log metrics (e.g., to console and/or W&B).
    interval : int
        Interval (in batches) at which to log training progress.
    scheduler : torch.optim.lr_scheduler._LRScheduler | None
        Optional learning rate scheduler.
    early_stopping_enabled : bool
        Whether to enable early stopping.
    early_stopping_patience : int
        Number of epochs to wait for improvement before stopping.
    early_stopping_min_delta : float
        Minimum change in validation loss to qualify as improvement.
    Returns
    -------
    Dict[str, float]
        A dictionary containing the best validation loss.
    """
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    if epochs == 0:
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        return {
            "best_val_loss": val_metrics["loss"],
            "stopped_early": False,
            "final_epoch": 0,
        }

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            log_fn,
            interval,
            scheduler,
            tokenizer,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, tokenizer)

        elapsed_time = time.time() - start_time

        log_obj = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "time_sec": elapsed_time,
        }
        log_fn(log_obj)

        # Check for improvement with min_delta threshold
        if val_metrics["loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            log_obj = {
                "message": f"Early stopping triggered after {epoch} epochs",
                "epochs_without_improvement": epochs_without_improvement,
                "best_val_loss": best_val_loss,
            }
            log_fn(log_obj)
            break

    assert best_state is not None, "Training loop failed to produce a best model state."
    model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val_loss,
        "stopped_early": early_stopping_enabled
        and epochs_without_improvement >= early_stopping_patience,
        "final_epoch": epoch,
    }
