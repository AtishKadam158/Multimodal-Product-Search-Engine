"""
Utility functions for file I/O, score normalization, and common helpers.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def normalize_scores_minmax(scores: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to a score array.

    Maps scores to [0, 1] range:
        normalized = (score - min) / (max - min)

    If all scores are identical, returns zeros to avoid division by zero.

    Args:
        scores: 1-D array of raw scores.

    Returns:
        Normalized scores in [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64)
    min_val = scores.min()
    max_val = scores.max()

    if max_val - min_val < 1e-12:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_embeddings(embeddings_path: Path, ids_path: Path):
    """
    Load pre-computed embeddings and their corresponding IDs from disk.

    Args:
        embeddings_path: Path to .npy file with embeddings.
        ids_path: Path to .npy file with matching IDs.

    Returns:
        Tuple of (embeddings: np.ndarray, ids: np.ndarray)
    """
    embeddings = np.load(str(embeddings_path))
    ids = np.load(str(ids_path))

    if len(embeddings) != len(ids):
        raise ValueError(
            f"Embedding/ID length mismatch: {len(embeddings)} embeddings vs {len(ids)} IDs"
        )

    return embeddings, ids
