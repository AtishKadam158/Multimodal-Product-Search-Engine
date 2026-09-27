"""
Score Fusion — combines scores from multiple retrieval systems.

Supports:
- Weighted late fusion (for multimodal: image_score + text_score)
- Hybrid fusion (for dense + BM25 combination)

Scores are normalized before fusion to prevent one system from dominating.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGE_WEIGHT, TEXT_WEIGHT, DENSE_WEIGHT, BM25_WEIGHT
from src.utils.io import normalize_scores_minmax


def weighted_fusion(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    weight_a: float,
    weight_b: float,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Combine results from two retrieval sources using weighted late fusion.

    Formula:
        final_score = weight_a * norm_score_a + weight_b * norm_score_b

    Scores are min-max normalized before combining.

    Args:
        results_a: Results from first source (list of dicts with 'id' and 'score').
        results_b: Results from second source.
        weight_a: Weight for first source.
        weight_b: Weight for second source.
        top_k: Number of final results to return.

    Returns:
        Fused and re-ranked list of result dicts.
    """
    # Collect all unique candidate IDs and their metadata
    candidates = {}  # id -> {metadata + scores}

    for r in results_a:
        rid = r["id"]
        if rid not in candidates:
            candidates[rid] = {
                "id": rid,
                "image_path": r.get("image_path", ""),
                "caption": r.get("caption", ""),
                "category": r.get("category", ""),
                "score_a": r["score"],
                "score_b": 0.0,
            }
        else:
            candidates[rid]["score_a"] = r["score"]

    for r in results_b:
        rid = r["id"]
        if rid not in candidates:
            candidates[rid] = {
                "id": rid,
                "image_path": r.get("image_path", ""),
                "caption": r.get("caption", ""),
                "category": r.get("category", ""),
                "score_a": 0.0,
                "score_b": r["score"],
            }
        else:
            candidates[rid]["score_b"] = r["score"]

    if not candidates:
        return []

    # Extract scores and normalize
    cand_list = list(candidates.values())
    scores_a = np.array([c["score_a"] for c in cand_list])
    scores_b = np.array([c["score_b"] for c in cand_list])

    norm_a = normalize_scores_minmax(scores_a)
    norm_b = normalize_scores_minmax(scores_b)

    # Compute fused scores
    fused = weight_a * norm_a + weight_b * norm_b

    # Sort by fused score (descending)
    ranked_indices = np.argsort(fused)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        c = cand_list[idx]
        results.append({
            "id": c["id"],
            "score": float(fused[idx]),
            "image_path": c["image_path"],
            "caption": c["caption"],
            "category": c["category"],
            "score_a": float(norm_a[idx]),
            "score_b": float(norm_b[idx]),
        })

    return results


def multimodal_fusion(
    image_results: List[Dict[str, Any]],
    text_results: List[Dict[str, Any]],
    image_weight: float = IMAGE_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fuse image-query and text-query retrieval results.

    Used for the Text + Image → Image search mode.

    Args:
        image_results: Results from image-query search.
        text_results: Results from text-query search.
        image_weight: Weight for image similarity (default 0.6).
        text_weight: Weight for text similarity (default 0.4).
        top_k: Number of final results.

    Returns:
        Fused results sorted by combined score.
    """
    return weighted_fusion(image_results, text_results, image_weight, text_weight, top_k)


def hybrid_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fuse dense retrieval and BM25 results for hybrid search.

    Formula:
        hybrid_score = alpha * norm_dense + beta * norm_bm25
        where alpha + beta = 1

    Args:
        dense_results: Results from dense (embedding) retrieval.
        bm25_results: Results from BM25 keyword retrieval.
        dense_weight: Alpha weight for dense scores.
        bm25_weight: Beta weight for BM25 scores.
        top_k: Number of final results.

    Returns:
        Hybrid-fused results.
    """
    return weighted_fusion(dense_results, bm25_results, dense_weight, bm25_weight, top_k)
