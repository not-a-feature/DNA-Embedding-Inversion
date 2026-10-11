"""Validate the generated embedding datasets on disk.

Opens every train/val/test HDF5 under data/hg38 and data/1000g, checks the row
count against what the pipeline is supposed to have produced, and samples a
window of rows to confirm the stored sequences are real DNA of the right length
and the embeddings are finite, non-degenerate numbers. Use it to decide whether
the existing (pre-rework) data is trustworthy or has to be regenerated.

This reads only a small sampled window per file, so it is cheap even on the
~100 GB Evo2 identification shards. Exits non-zero if any file FAILs.

Usage
-----
    python scripts/validate_data.py                 # validate everything
    python scripts/validate_data.py data/hg38       # restrict to a subdir
    python scripts/validate_data.py --sample 8192   # rows sampled per file
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np

# Count-based split of the per-length hg38 cohort: val/test are a fixed 50k each
# and train is everything else, so train = 900k for the 1M base lengths.
BASE_TRAIN, BASE_VAL, BASE_TEST = 900_000, 50_000, 50_000
# Extra TRAIN rows at the longer lengths (the per-length CSV is sized 1M+EXTRA in a
# single pass; val/test stay 50k, so only train grows).
EXTRA = {25: 1_000_000, 30: 1_000_000, 35: 1_000_000, 40: 1_000_000, 45: 1_000_000,
         50: 2_000_000, 60: 2_000_000, 70: 2_000_000, 80: 2_000_000,
         90: 3_000_000, 100: 3_000_000}
# Per-token generation (length 100 only) embeds a small fixed cohort.
PER_TOKEN_TOTAL = 10_000
VALID_BASES = set("ACGTN")
FMS = ("dnabert2", "evo2", "ntv2")
# Rows read per IO chunk when streaming (--full) or per sampled window.
CHUNK = 50_000


def parse_name(path):
    """Return (kind, fm, length, split, mode) parsed from a dataset filename."""
    b = os.path.basename(path)
    m = re.match(r"(train|val|test)_(\w+?)_(\d+)_hg38_per_token\.h5$", b)
    if m:
        return ("per_token", m.group(2), int(m.group(3)), m.group(1), None)
    m = re.match(r"(train|val|test)_(\w+?)_(\d+)_hg38\.h5$", b)
    if m:
        return ("hg38_mean", m.group(2), int(m.group(3)), m.group(1), None)
    m = re.match(r"test_(\w+?)_(\d+)_1000g_id_(random|snp_enriched)\.h5$", b)
    if m:
        return ("id", m.group(1), int(m.group(2)), "test", m.group(3))
    m = re.match(r"test_(\w+?)_(\d+)_1000g\.h5$", b)
    if m:
        return ("ood", m.group(1), int(m.group(2)), "test", None)
    return (None, None, None, None, None)


def expected_rows(kind, length, split):
    """Expected row count for the deterministic hg38 datasets, else None."""
    if kind == "hg38_mean":
        if split == "train":
            return BASE_TRAIN + EXTRA.get(length, 0)
        return BASE_VAL if split == "val" else BASE_TEST
    return None  # per_token split / id / ood counts checked structurally below


def expected_id_rows(path):
    """Expected identification rows = cohort * loci_written * 2 alleles.

    The id H5 filename carries no N/K, so resolve the cohort/loci from the
    matching CSV manifest. Both the K1000 smoke and the K4096 full-run manifests
    exist for each mode; the full run (max num_loci) is the one the H5 came from.
    """
    mode = "snp_enriched" if "snp_enriched" in os.path.basename(path) else "random"
    cand = [c for c in glob.glob(os.path.join(os.path.dirname(path), "1000g_id_*.manifest.json"))
            if mode in os.path.basename(c)]
    if not cand:
        return None
    best = None
    for c in cand:
        with open(c) as fh:
            man = json.load(fh)
        if best is None or man["num_loci"] > best["num_loci"]:
            best = man
    return best["cohort_size"] * len(best["loci"]) * 2


def validate(path, sample):
    """Validate one file; return (status, dim, notes) where status in PASS/WARN/FAIL."""
    kind, fm, length, split, mode = parse_name(path)
    notes = []
    if kind is None:
        return ("SKIP", None, ["unrecognized filename"])

    with h5py.File(path, "r") as f:
        for key in ("sequences", "embeddings"):
            assert key in f, f"missing dataset '{key}'"
        seqs, embs = f["sequences"], f["embeddings"]
        n = len(seqs)
        if n != embs.shape[0]:
            return ("FAIL", None, [f"seq/emb row mismatch {n} vs {embs.shape[0]}"])
        dim = int(embs.shape[-1])
        is_pt = kind == "per_token"
        if is_pt and embs.ndim != 3:
            notes.append(f"per_token embeddings ndim={embs.ndim} (expected 3)")
        if not is_pt and embs.ndim != 2:
            notes.append(f"mean embeddings ndim={embs.ndim} (expected 2)")

        # --- row count ---
        exp = expected_rows(kind, length, split)
        status = "PASS"
        if exp is not None and n != exp:
            status = "FAIL"
            notes.append(f"rows={n:,} expected={exp:,}")
        if kind == "per_token" and split == "train" and not (0 < n <= PER_TOKEN_TOTAL):
            status = "FAIL"; notes.append(f"per_token train rows={n:,} outside (0,{PER_TOKEN_TOTAL}]")
        if kind == "id":
            exp_id = expected_id_rows(path)
            if exp_id is not None and n != exp_id:
                status = "FAIL"
                notes.append(f"id rows={n:,} != manifest cohort*loci*2={exp_id:,}")
        if n == 0:
            return ("FAIL", dim, ["empty file (0 rows)"])

        # --- sample a contiguous window ---
        s = min(sample, n)
        start = 0 if n == s else int(np.random.default_rng(0).integers(0, n - s))
        seq_sample = seqs[start:start + s]
        emb_sample = np.asarray(embs[start:start + s])

        # --- sequence sanity ---
        decoded = [x.decode() if isinstance(x, bytes) else str(x) for x in seq_sample]
        bad_char = sum(1 for q in decoded if not set(q) <= VALID_BASES)
        bad_len = sum(1 for q in decoded if len(q) != length)
        n_frac = np.mean([q.count("N") / max(len(q), 1) for q in decoded])
        uniq = len(set(decoded)) / len(decoded)
        joined = "".join(decoded)
        gc = (joined.count("G") + joined.count("C")) / max(len(joined), 1)
        if bad_char:
            status = "FAIL"; notes.append(f"{bad_char}/{s} seqs have non-ACGTN chars")
        if bad_len:
            status = "FAIL"; notes.append(f"{bad_len}/{s} seqs wrong length (want {length})")
        if n_frac > 0.05:
            status = "FAIL" if status != "FAIL" else status
            notes.append(f"N-fraction {n_frac:.2%} high")
        if not (0.25 <= gc <= 0.75):
            notes.append(f"GC={gc:.2%} unusual")

        # --- embedding sanity ---
        # Per-token shards store variable-length rows as an object array (BPE
        # token counts differ per sequence), so reduce row-by-row; mean shards
        # are rectangular and reduce in one shot.
        if emb_sample.dtype == object:
            rows = [np.asarray(r, dtype=np.float32).ravel() for r in emb_sample]
            finite = all(np.isfinite(r).all() for r in rows)
            zero_rows = sum(1 for r in rows if np.abs(r).sum() == 0)
            allv = np.concatenate(rows)
            row_std = float(allv.std())
            flat = allv.reshape(1, -1)
        else:
            flat = emb_sample.reshape(emb_sample.shape[0], -1)
            finite = np.isfinite(flat).all()
            zero_rows = int((np.abs(flat).sum(axis=1) == 0).sum())
            row_std = float(flat.std(axis=0).mean())
        if not finite:
            status = "FAIL"; notes.append("non-finite values (NaN/Inf)")
        if zero_rows:
            status = "FAIL"; notes.append(f"{zero_rows}/{s} all-zero embedding rows")
        if row_std == 0.0:
            status = "FAIL"; notes.append("zero variance across rows (collapsed)")

        # --- stored stats consistency (train mean files only) ---
        if kind == "hg38_mean" and split == "train":
            for a in ("emb_min", "emb_max", "emb_mean", "emb_std"):
                assert a in f.attrs, f"missing stat attr '{a}'"
            stored_mean = float(f.attrs["emb_mean"])
            stored_std = float(f.attrs["emb_std"])
            sm = float(flat.mean())
            tol = 0.1 * (abs(stored_mean) + stored_std + 1e-6)
            if abs(sm - stored_mean) > max(tol, 0.05):
                notes.append(f"sample mean {sm:.4f} vs stored emb_mean {stored_mean:.4f}")

        notes.append(f"gc={gc:.0%} uniq={uniq:.0%} rowstd={row_std:.3f}")
        return (status, dim, notes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", default=["data/hg38", "data/1000g"],
                    help="directories to scan (default: data/hg38 data/1000g)")
    ap.add_argument("--sample", type=int, default=4096, help="rows sampled per file")
    args = ap.parse_args()

    files = []
    for r in args.roots:
        files += glob.glob(os.path.join(r, "*.h5"))
    files = sorted(set(files))
    if not files:
        print("No .h5 files found."); sys.exit(1)

    dims_by_fm = {}
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    fails = []
    print(f"Validating {len(files)} files (sample={args.sample} rows each)\n")
    print(f"{'status':6} {'file':52} {'dim':>5}  notes")
    print("-" * 110)
    for p in files:
        try:
            status, dim, notes = validate(p, args.sample)
        except Exception as e:  # structural assertion -> hard fail, keep going
            status, dim, notes = "FAIL", None, [f"{type(e).__name__}: {e}"]
        counts[status] += 1
        if status == "FAIL":
            fails.append(p)
        _, fm, _, _, _ = parse_name(p)
        if dim is not None and fm in FMS:
            dims_by_fm.setdefault(fm, {}).setdefault(dim, 0)
            dims_by_fm[fm][dim] += 1
        print(f"{status:6} {os.path.basename(p):52} {str(dim):>5}  {'; '.join(notes)}")

    print("\n=== embedding dim consistency per FM ===")
    for fm, dims in sorted(dims_by_fm.items()):
        flag = "" if len(dims) == 1 else "  <-- INCONSISTENT"
        print(f"  {fm:10} {dict(sorted(dims.items()))}{flag}")

    print(f"\nSummary: PASS={counts['PASS']} WARN={counts['WARN']} "
          f"FAIL={counts['FAIL']} SKIP={counts['SKIP']}")
    if fails:
        print("\nFAILED files:")
        for p in fails:
            print(f"  {p}")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
