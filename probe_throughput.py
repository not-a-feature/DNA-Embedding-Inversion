"""Measure end-to-end training throughput vs batch size on the GPU.

The multi-length training was data-bound: batch_size=32 with num_workers=0 ran at
~880 samples/sec because every sample was a random HDF5 row read serialised in the
main process. With per-process handles and num_workers=16 the data path sustains
~29k samples/sec at batch_size=32, so the batch size no longer buys throughput
there -- but that only holds if the GPU can consume what the loader delivers.

This probe times real forward/backward/step iterations (loader included) so the
reported number is what training would actually reach. Read it as:

  * throughput flat from 32 upwards  -> keep batch_size=32; it costs nothing and
    preserves the most optimizer steps, which the undertrained mean runs need.
  * throughput climbing with batch   -> the run is GPU-bound at 32 and a larger
    batch buys wall time, paid for with proportionally fewer updates.

Run on one GPU (stage 0 of calibrate.sbatch).
"""

from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time

import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import compute_pooled_train_stats, create_dataset
from src.model.query_decoder import QueryDecoderReconstructor
from src.tokenizers import CharacterTokenizer
from src.utils import variable_length_collate

BATCH_SIZES = [32, 64, 128, 256, 512]
NUM_WORKERS = 16
STEPS = 120
# Evo2 is the widest input (4096-d) and therefore the least favourable case for
# small batches; if batch_size=32 saturates the GPU here it saturates it for the
# 768-d / 1024-d models too.
TRAIN_H5 = "data/hg38/train_evo2_50_hg38.h5"


def main() -> None:
    assert torch.cuda.is_available(), "probe_throughput.py needs a GPU"
    device = torch.device("cuda:0")

    h5 = h5py.File(TRAIN_H5, "r", swmr=True)
    stats = compute_pooled_train_stats([h5])
    tokenizer = CharacterTokenizer()
    dataset = create_dataset(
        data=h5,
        mode="mean",
        tokenizer=tokenizer,
        embedding_dim=4096,
        seq_length=50,
        normalization_stats=stats,
        normalization_method="standard",
        data_is_mean=True,
    )

    print(f"{'batch':>6} {'samples/sec':>12} {'steps/sec':>10} {'est. h/epoch (31.6M)':>22}")
    for batch_size in BATCH_SIZES:
        model = QueryDecoderReconstructor(
            input_dim=4096,
            mode="mean",
            seq_length=51,
            output_dim=tokenizer.vocab_size,
            d_model=384,
            nhead=8,
            num_layers=6,
            dim_feedforward=1536,
            dropout=0.1,
            n_context_tokens=16,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-6)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=variable_length_collate,
        )

        it = iter(loader)
        for _ in range(10):  # warm up workers, allocator and cudnn autotune
            emb, seq = next(it)
            out = model(emb.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            loss_fn(logits.permute(0, 2, 1)[:, :, : seq.size(1)], seq.to(device))
        torch.cuda.synchronize()

        start = time.time()
        seen = 0
        for _ in range(STEPS):
            emb, seq = next(it)
            emb, seq = emb.to(device, non_blocking=True), seq.to(device, non_blocking=True)
            out = model(emb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(logits.permute(0, 2, 1)[:, :, : seq.size(1)], seq)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += emb.shape[0]
        torch.cuda.synchronize()

        elapsed = time.time() - start
        rate = seen / elapsed
        print(f"{batch_size:>6} {rate:>12.1f} {STEPS / elapsed:>10.1f} {31_600_000 / rate / 3600:>22.2f}")
        del loader, model, optimizer


if __name__ == "__main__":
    main()
