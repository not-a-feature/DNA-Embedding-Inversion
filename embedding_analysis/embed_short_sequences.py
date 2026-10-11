"""Quick script to embed all combinations of short DNA sequences and compute pairwise distances."""

import itertools
import numpy as np
import torch
from evo2 import Evo2
from scipy.spatial.distance import pdist, squareform
import pandas as pd


NUCLEOTIDES = ["A", "C", "G", "T"]


def embed_sequence(model: Evo2, sequence: str, layer_name: str, device: str) -> np.ndarray:
    """Extract embeddings for a DNA sequence and return the mean."""
    tokens = model.tokenizer.tokenize(sequence)
    assert tokens, "Sequence must tokenize to at least one token"

    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])

    assert layer_name in embeddings, f"Layer '{layer_name}' not found in embeddings"

    tensor = embeddings[layer_name][0].detach().float().cpu().numpy()
    # Return mean over sequence length
    return tensor.mean(axis=0)


def generate_all_sequences(length: int) -> list:
    """Generate all possible DNA sequences of given length."""
    return ["".join(seq) for seq in itertools.product(NUCLEOTIDES, repeat=length)]


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "evo2_7b"
    layer_name = "blocks.26.mlp.l3"  # Last layer

    print(f"Using device: {device}")
    print(f"Loading Evo2 model from {checkpoint}...")

    # Load model (Evo2 handles device internally)
    model = Evo2(checkpoint)

    # Process length 2 and 3
    for length in [2, 3]:
        print(f"\n{'='*60}")
        print(f"Processing sequences of length {length}")
        print(f"{'='*60}")

        # Generate all sequences
        sequences = generate_all_sequences(length)
        print(f"Total sequences: {len(sequences)}")

        # Embed all sequences
        print("Generating embeddings...")
        embeddings = []
        for seq in sequences:
            emb = embed_sequence(model, seq, layer_name, device)
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        print(f"Embeddings shape: {embeddings.shape}")

        # Compute pairwise Euclidean distances
        distances = squareform(pdist(embeddings, metric="euclidean"))

        # Create pretty DataFrame
        df = pd.DataFrame(distances, index=sequences, columns=sequences)

        print(f"\nPairwise Euclidean distances (length {length}):")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))

        # Print some statistics
        print(f"\nStatistics:")
        print(f"  Min distance: {distances[distances > 0].min():.4f}")
        print(f"  Max distance: {distances.max():.4f}")
        print(f"  Mean distance: {distances[distances > 0].mean():.4f}")
        print(f"  Std distance: {distances[distances > 0].std():.4f}")


if __name__ == "__main__":
    main()
