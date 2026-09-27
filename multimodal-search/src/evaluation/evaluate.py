"""
Evaluation Runner — compares BM25, Dense, Hybrid, and Hybrid+Reranker
retrieval methods against ground-truth queries.

Generates a results table and saves to results/evaluation.csv.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import QUERIES_PATH, RESULTS_DIR, TOP_K
from src.evaluation.metrics import evaluate_queries


def load_queries(queries_path: str = None) -> List[Dict[str, Any]]:
    """
    Load evaluation queries from queries.json.

    Each query has:
        - "query": text string
        - "relevant_ids": list of relevant COCO image IDs

    Returns:
        List of query dicts.
    """
    path = queries_path or str(QUERIES_PATH)
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} evaluation queries from {path}")
    return queries


def run_evaluation(
    dense_retriever=None,
    bm25_retriever=None,
    hybrid_retriever=None,
    reranker=None,
    queries: List[Dict[str, Any]] = None,
    k: int = TOP_K,
    rerank_top_n: int = 50,
) -> pd.DataFrame:
    """
    Run evaluation across all retrieval methods.

    Compares: BM25, Dense, Hybrid, Hybrid + Reranker

    Args:
        dense_retriever: DenseRetriever instance.
        bm25_retriever: BM25Retriever instance.
        hybrid_retriever: HybridRetriever instance.
        reranker: Reranker instance (optional).
        queries: List of query dicts with 'query' and 'relevant_ids'.
        k: Cutoff for metrics (e.g., 10).
        rerank_top_n: Number of candidates to pass to reranker.

    Returns:
        DataFrame with evaluation results per method.
    """
    if queries is None:
        queries = load_queries()

    methods = {}

    # --- BM25 ---
    if bm25_retriever is not None:
        print("\nEvaluating: BM25 ...")
        bm25_retrieved = []
        for q in queries:
            results = bm25_retriever.search(q["query"], top_k=k)
            bm25_retrieved.append([r["id"] for r in results])

        all_relevant = [q["relevant_ids"] for q in queries]
        methods["BM25"] = evaluate_queries(bm25_retrieved, all_relevant, k)

    # --- Dense ---
    if dense_retriever is not None:
        print("Evaluating: Dense Retrieval ...")
        dense_retrieved = []
        for q in queries:
            results = dense_retriever.search_text(q["query"], top_k=k)
            dense_retrieved.append([r["id"] for r in results])

        all_relevant = [q["relevant_ids"] for q in queries]
        methods["Dense"] = evaluate_queries(dense_retrieved, all_relevant, k)

    # --- Hybrid ---
    if hybrid_retriever is not None:
        print("Evaluating: Hybrid Retrieval ...")
        hybrid_retrieved = []
        for q in queries:
            results = hybrid_retriever.search(q["query"], top_k=k)
            hybrid_retrieved.append([r["id"] for r in results])

        all_relevant = [q["relevant_ids"] for q in queries]
        methods["Hybrid"] = evaluate_queries(hybrid_retrieved, all_relevant, k)

    # --- Hybrid + Reranker ---
    if hybrid_retriever is not None and reranker is not None:
        print("Evaluating: Hybrid + Reranker ...")
        reranked_retrieved = []
        for q in queries:
            # Get more candidates for reranking
            candidates = hybrid_retriever.search(q["query"], top_k=rerank_top_n)
            reranked = reranker.rerank(q["query"], candidates, top_k=k)
            reranked_retrieved.append([r["id"] for r in reranked])

        all_relevant = [q["relevant_ids"] for q in queries]
        methods["Hybrid + Reranker"] = evaluate_queries(reranked_retrieved, all_relevant, k)

    # Build results table
    rows = []
    for method_name, scores in methods.items():
        rows.append({
            "Method": method_name,
            f"Recall@{k}": round(scores["recall"], 4),
            f"Precision@{k}": round(scores["precision"], 4),
            f"MRR@{k}": round(scores["mrr"], 4),
            f"NDCG@{k}": round(scores["ndcg"], 4),
            f"HitRate@{k}": round(scores["hit_rate"], 4),
        })

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 70}")
    print(f"Evaluation Results (K={k}, {len(queries)} queries)")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))

    return df


def save_results(df: pd.DataFrame, output_dir: str = None) -> str:
    """Save evaluation results to CSV."""
    out_dir = Path(output_dir or str(RESULTS_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "evaluation.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    return str(output_path)
