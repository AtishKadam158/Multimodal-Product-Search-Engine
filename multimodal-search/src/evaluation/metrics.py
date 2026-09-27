"""
Evaluation Metrics — Recall@K, Precision@K, MRR, NDCG@K, Hit Rate@K.

All metrics accept:
    - retrieved_ids: ordered list of retrieved document IDs
    - relevant_ids: set/list of ground-truth relevant document IDs
"""

import numpy as np
from typing import List, Set, Union


def recall_at_k(
    retrieved_ids: List[int],
    relevant_ids: Union[List[int], Set[int]],
    k: int,
) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

    What fraction of the relevant documents were found in the top-K?

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Recall score in [0, 1].
    """
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])
    hits = len(relevant_set & retrieved_top_k)
    return hits / len(relevant_set)


def precision_at_k(
    retrieved_ids: List[int],
    relevant_ids: Union[List[int], Set[int]],
    k: int,
) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:k]| / K

    What fraction of the top-K results are actually relevant?

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Precision score in [0, 1].
    """
    relevant_set = set(relevant_ids)
    if k == 0:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])
    hits = len(relevant_set & retrieved_top_k)
    return hits / k


def mrr(
    retrieved_ids: List[int],
    relevant_ids: Union[List[int], Set[int]],
) -> float:
    """
    Mean Reciprocal Rank (for a single query).

    MRR = 1 / rank_of_first_relevant_document

    If no relevant document is found, returns 0.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Ground-truth relevant document IDs.

    Returns:
        Reciprocal rank in [0, 1].
    """
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: List[int],
    relevant_ids: Union[List[int], Set[int]],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Uses binary relevance: 1 if document is relevant, 0 otherwise.

    DCG@K  = Σ (rel_i / log2(i + 1))  for i = 1..K
    IDCG@K = Σ (1 / log2(i + 1))      for i = 1..min(K, |relevant|)
    NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        NDCG score in [0, 1].
    """
    relevant_set = set(relevant_ids)
    if not relevant_set or k == 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed, rank is 1-indexed

    # Ideal DCG: all relevant docs ranked at the top
    ideal_k = min(k, len(relevant_set))
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(
    retrieved_ids: List[int],
    relevant_ids: Union[List[int], Set[int]],
    k: int,
) -> float:
    """
    Hit Rate@K = 1 if any relevant document is in top-K, else 0.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        1.0 or 0.0
    """
    relevant_set = set(relevant_ids)
    retrieved_top_k = set(retrieved_ids[:k])
    return 1.0 if (relevant_set & retrieved_top_k) else 0.0


def evaluate_single_query(
    retrieved_ids: List[int],
    relevant_ids: Union[List[int], Set[int]],
    k: int = 10,
) -> dict:
    """
    Compute all metrics for a single query at a given K.

    Returns:
        Dict with keys: recall, precision, mrr, ndcg, hit_rate
    """
    return {
        "recall": recall_at_k(retrieved_ids, relevant_ids, k),
        "precision": precision_at_k(retrieved_ids, relevant_ids, k),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, k),
        "hit_rate": hit_rate_at_k(retrieved_ids, relevant_ids, k),
    }


def evaluate_queries(
    all_retrieved: List[List[int]],
    all_relevant: List[Union[List[int], Set[int]]],
    k: int = 10,
) -> dict:
    """
    Compute average metrics across multiple queries.

    Args:
        all_retrieved: List of retrieved ID lists (one per query).
        all_relevant: List of relevant ID sets (one per query).
        k: Cutoff rank.

    Returns:
        Dict with mean values for each metric.
    """
    metrics = {"recall": [], "precision": [], "mrr": [], "ndcg": [], "hit_rate": []}

    for retrieved, relevant in zip(all_retrieved, all_relevant):
        result = evaluate_single_query(retrieved, relevant, k)
        for key in metrics:
            metrics[key].append(result[key])

    return {key: float(np.mean(values)) for key, values in metrics.items()}
