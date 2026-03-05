"""
DNA Embedding Inversion Attack - Training Entry Point
"""

from __future__ import annotations

import os

# Disable tokenizer parallelism to avoid deadlocks when using multiple workers in DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil

from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader
import hydra
import logging

from src.utils import (
    set_determinism,
    maybe_init_wandb,
    log_factory,
    save_json,
    dynamic_import_class,
    variable_length_collate,
)
from src.data import (
    load_split_embeddings,
    create_dataset,
)
from src.train import fit, evaluate
from src.tokenizers import CharacterTokenizer


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: D401
    """Execute the DNA embedding inversion pipeline.

    This function orchestrates the entire workflow, from configuration loading
    and determinism setup to data loading, model training, and artifact
    persistence. It adheres to a strict, fail-fast philosophy.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra configuration object, composed from YAML files and command-line
        overrides. It contains all settings for the run.

    """
    logger = logging.getLogger(__name__)
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    # 1. Determinism and environment setup
    set_determinism(cfg.train.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.train.device)

    # 2. Validate and get mode from model config
    mode = cfg.model.mode
    assert mode in [
        "per_token",
        "mean",
    ], f"Invalid mode: {mode}. Must be 'per_token' or 'mean'"
    logger.info(f"Training mode: {mode}")

    # 3. Check for existing run
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if os.path.exists(os.path.join(output_dir, "model.pt")):
        logger.info(f"Model already exists in {output_dir}. Skipping training.")
        return

    # 3b. Setup Tokenizer — always character-level (A=0, C=1, G=2, T=3)
    tokenizer = CharacterTokenizer()
    logger.info(f"Initialized CharacterTokenizer, vocab_size={tokenizer.vocab_size}")

    # 4. Dynamically import model class from configured model file
    model_file_path = os.path.join(
        os.path.dirname(__file__),
        "src",
        cfg.model.model_file_name,
    )

    # Determine model class name based on model type
    model_type = cfg.model.model_type
    if model_type == "encoder":
        model_class_name = "EncoderReconstructor"
    elif model_type == "decoder":
        model_class_name = "DecoderReconstructor"
    elif model_type == "knn":
        model_class_name = "KNNReconstructor"
    elif model_type == "resnet":
        model_class_name = "ResNetReconstructor"

    elif mode == "mean":
        model_class_name = "SequenceMeanReconstructor"
    else:
        model_class_name = "SequenceReconstructor"

    ModelClass = dynamic_import_class(model_file_path, model_class_name)

    # 5. Load data with HDF5 for true lazy loading
    data_dict, counts_dict, train_stats = load_split_embeddings(cfg.data)
    logger.info(f"Loaded - train: {counts_dict['train']}, val: {counts_dict['val']}")

    data_is_mean = cfg.data.get("mean", False)

    # Log training set normalization statistics (used for all splits to avoid data leakage)
    if train_stats:
        logger.info(
            f"Training set embedding stats (applied to all splits) - "
            f"min: {train_stats['min']:.4f}, max: {train_stats['max']:.4f}, "
            f"mean: {train_stats['mean']:.4f}, std: {train_stats['std']:.4f}"
        )

    # 6. Build DataLoaders with lazy-loaded datasets
    # All datasets use training set statistics for normalization to prevent data leakage
    loaders = {}
    for split in ["train", "val"]:
        max_samples = None
        if split == "val":
            max_samples = cfg.train.max_val_samples

        dataset = create_dataset(
            data_dict[split],
            mode,
            tokenizer,
            cfg.data.embedding_dim,
            cfg.data.seq_length,
            normalization_stats=train_stats if train_stats else None,
            normalization_method=cfg.data.normalization_method,
            data_is_mean=data_is_mean,
            subset_fraction=cfg.data.subset_fraction,
            max_samples=max_samples,
        )

        use_workers = cfg.optim.num_workers > 0
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.optim.batch_size,
            shuffle=(split == "train" and cfg.data.shuffle),
            num_workers=cfg.optim.num_workers,
            pin_memory=True,
            persistent_workers=use_workers,
            collate_fn=variable_length_collate,
        )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    logger.info(f"DataLoaders created - train: {len(train_loader)}, val: {len(val_loader)} batches")

    # 7. Instantiate model, optimizer, and loss function
    # Character tokenizer: 1 token per nucleotide, so seq_length == number of tokens
    seq_length = cfg.data.seq_length

    # Build model kwargs based on model type and mode
    if model_type == "encoder" or model_type == "decoder":
        # Encoder/Decoder calculates output_dim internally from seq_length * output_dim
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "hidden_dims": cfg.model.hidden_dims,
            "mode": mode,
            "seq_length": seq_length,
            "output_dim": tokenizer.vocab_size,
            "d_model": cfg.model.d_model,
            "nhead": cfg.model.nhead,
            "num_layers": cfg.model.num_layers,
            "dim_feedforward": cfg.model.dim_feedforward,
            "dropout": cfg.model.dropout,
        }
    elif model_type == "knn":
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "output_dim": tokenizer.vocab_size,
            "k": cfg.model.k,
        }

    elif model_type == "resnet":
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "mode": mode,
            "seq_length": seq_length,
            "output_dim": tokenizer.vocab_size,
            "d_model": cfg.model.d_model,
            "n_blocks": cfg.model.n_blocks,
            "kernel_size": cfg.model.kernel_size,
            "dropout": cfg.model.dropout,
        }

    elif mode == "mean":
        # Mean mode MLP calculates output_dim from seq_length * output_dim
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "hidden_dims": cfg.model.hidden_dims,
            "seq_length": seq_length,
            "output_dim": tokenizer.vocab_size,
            "dropout": cfg.model.dropout,
        }
    else:
        # Per-nucleotide mode uses output_dim as output_dim
        model_kwargs = {
            "input_dim": cfg.data.embedding_dim,
            "hidden_dims": cfg.model.hidden_dims,
            "output_dim": tokenizer.vocab_size,
            "dropout": cfg.model.dropout,
        }

    model = ModelClass(**model_kwargs).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if model_type == "knn":
        logger.info(f"Fitting {model_type} model with training data...")
        model.fit(train_loader)

        logger.info(f"Evaluating {model_type} model...")
        # Note: evaluate needs update to handle tokenizer if it uses decode_onehot_to_sequence
        val_metrics = evaluate(model, val_loader, loss_fn, device, tokenizer=tokenizer)
        results = {
            "best_val_loss": val_metrics["loss"],
            "stopped_early": False,
            "final_epoch": 0,
        }

        # Initialize wandb for KNN/Chroma if needed (for consistency)
        wandb_run = maybe_init_wandb(cfg)
    else:

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )

        # Create learning rate scheduler if enabled
        scheduler = None
        if cfg.optim.use_scheduler:
            total_steps = len(train_loader) * cfg.optim.epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.optim.scheduler.max_lr,
                total_steps=total_steps,
                pct_start=cfg.optim.scheduler.pct_start,
                anneal_strategy=cfg.optim.scheduler.anneal_strategy,
                div_factor=cfg.optim.scheduler.div_factor,
                final_div_factor=cfg.optim.scheduler.final_div_factor,
            )
            logger.info(
                f"OneCycleLR scheduler enabled - max_lr: {cfg.optim.scheduler.max_lr}, "
                f"total_steps: {total_steps}, pct_start: {cfg.optim.scheduler.pct_start}"
            )

        # 8. Set up logging and execute training
        wandb_run = maybe_init_wandb(cfg)
        log_fn = log_factory(logger, wandb_run)

        results = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=cfg.optim.epochs,
            device=device,
            log_fn=log_fn,
            tokenizer=tokenizer,
            interval=cfg.train.log_interval,
            scheduler=scheduler,
            early_stopping_enabled=cfg.train.early_stopping.enabled,
            early_stopping_patience=cfg.train.early_stopping.patience,
            early_stopping_min_delta=cfg.train.early_stopping.min_delta,
        )

    logger.info(
        "Final results: " + ", ".join(f"{k}={v:.4f}" for k, v in results.items())
    )  # 9. Persist artifacts

    # Copy model file to output directory for reproducibility
    model_source = os.path.join(os.path.dirname(__file__), "src", cfg.model.model_file_name)
    model_dest = os.path.join(output_dir, "model.py")
    shutil.copy2(model_source, model_dest)
    logger.info(f"Copied {cfg.model.model_file_name} to {model_dest}")

    model_path = os.path.join(output_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "results": results,
            "input_dim": cfg.data.embedding_dim,
            "output_dim": tokenizer.vocab_size,
            "mode": mode,
            "effective_seq_length": seq_length,
            "tokenizer_type": "char",
            "tokenizer_model": None,
        },
        model_path,
    )
    logger.info(f"Saved model checkpoint to {model_path}")

    # Also store results separately for easy inspection
    save_json(results, os.path.join(output_dir, "results.json"))
    logger.info(f"Saved results to {output_dir}")

    if wandb_run:
        wandb_run.summary.update(results)
        wandb_run.finish()


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
