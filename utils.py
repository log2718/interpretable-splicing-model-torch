"""Utility helpers for generating and encoding DNA sequences."""

import numpy as np
import random
random.seed(42)

def add_flanking(seqs):
    """Add the fixed flanking context expected by the model.

    Args:
        seqs: Iterable of exon sequences as strings.

    Returns:
        A list of sequences with the canonical left and right flanks appended.
    """
    left, right = 'CATCCAGGTT', 'CAGGTCTGAC'
    return [left + s + right for s in seqs]

def str_to_vector(s, alphabet="ACGT"):
    """Convert a nucleotide string to a one-hot matrix.

    Args:
        s: Input string to encode.
        alphabet: Ordered alphabet used to build the one-hot basis.

    Returns:
        A NumPy array of shape ``(len(s), len(alphabet))``.

    Raises:
        KeyError: If ``s`` contains a character not present in ``alphabet``.
    """
    idx = {c:i for i,c in enumerate(alphabet)}
    a = np.eye(len(alphabet))
    return a[[idx[c] for c in s]]

def one_hot_batch(seqs):
    """One-hot encode a batch of DNA or RNA sequences.

    Args:
        seqs: Iterable of nucleotide sequences. Any ``"U"`` characters are
            converted to ``"T"`` before encoding.

    Returns:
        A NumPy array with shape ``(batch_size, sequence_length, 4)``.
    """
    # DNA only; convert U->T if present
    seqs = [s.replace("U", "T") for s in seqs]
    return np.array([str_to_vector(s) for s in seqs])  # shape: [N, L, 4]

# simulate random ACGT sequences of specified length
def generate_random_exon(length):
    """Generate a random exon sequence.

    Args:
        length: Number of nucleotides to generate.

    Returns:
        A random DNA string of length ``length``.
    """
    return ''.join(random.choice('ACGT') for _ in range(length))
