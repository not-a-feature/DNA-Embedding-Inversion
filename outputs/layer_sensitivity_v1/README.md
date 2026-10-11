# Layer sensitivity

Fresh LCQDs; frozen foundation models. Per length: 100k train / 5k validation /
10k test sequences. Seed 42, 20 epochs, batch 32; best validation checkpoint.
Single-seed, fixed-budget results.

## Mean Levenshtein similarity

| Model | Layer/block | 25 nt | 50 nt | 100 nt |
| --- | ---: | ---: | ---: | ---: |
| dnabert2 | 3 | 0.718 | 0.597 | 0.530 |
| dnabert2 | 6 | 0.596 | 0.538 | 0.508 |
| dnabert2 | 9 | 0.540 | 0.501 | 0.488 |
| dnabert2 | 12 | 0.496 | 0.475 | 0.480 |
| ntv2 | 7 | 0.931 | 0.705 | 0.588 |
| ntv2 | 15 | 0.957 | 0.805 | 0.655 |
| ntv2 | 22 | 0.946 | 0.786 | 0.659 |
| ntv2 | 29 | 0.908 | 0.697 | 0.572 |
| evo2 | 7 | 0.570 | 0.475 | 0.418 |
| evo2 | 15 | 0.608 | 0.483 | 0.419 |
| evo2 | 26 | 0.729 | 0.534 | 0.437 |
| evo2 | 31 | 0.371 | 0.365 | 0.312 |

Paper layers: DNABERT-2 **12**, NTv2 **29**, Evo 2 **26**.
DNABERT-2/NTv2 indices are one-based; Evo 2 uses zero-based `blocks.{index}.mlp.l3`.

## Reproduce

Requires the benchmark environment, sequence-disjoint hg38 corpus, GPU and Evo 2 runtime.
Run from the repository root; indices follow table order:

```bash
python scripts/layer_sensitivity.py prepare --workspace outputs/layer_reproduction
for index in $(seq 0 11); do
  python scripts/layer_sensitivity.py run --index "$index" --workspace outputs/layer_reproduction
done
```

Includes metrics, training summaries, original-run manifests and the dataset audit.
