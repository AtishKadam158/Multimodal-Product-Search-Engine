"""
Evaluation script — compare BM25, Dense, Hybrid, and Hybrid+Reranker.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --k 10 --no-reranker
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TOP_K, RERANK_TOP_N


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods")
    parser.add_argument("--k", type=int, default=TOP_K, help="Cutoff K for metrics")
    parser.add_argument("--no-reranker", action="store_true", help="Skip reranker evaluation")
    parser.add_argument("--queries", type=str, default=None, help="Path to queries.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Retrieval Method Evaluation")
    print("=" * 60)

    # Initialize retrievers
    print("\nLoading components ...")

    from src.models.multimodal_encoder import MultimodalEncoder
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.evaluation.evaluate import load_queries, run_evaluation, save_results

    # Load model once
    encoder = MultimodalEncoder()

    # Initialize retrievers
    dense = DenseRetriever(encoder=encoder)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    # Optional reranker
    reranker = None
    if not args.no_reranker:
        try:
            from src.ranking.reranker import Reranker
            reranker = Reranker()
        except Exception as e:
            print(f"Warning: Could not load reranker: {e}")
            print("Continuing without reranker.")

    # Load queries
    queries = load_queries(args.queries)

    # Run evaluation
    results_df = run_evaluation(
        dense_retriever=dense,
        bm25_retriever=bm25,
        hybrid_retriever=hybrid,
        reranker=reranker,
        queries=queries,
        k=args.k,
        rerank_top_n=RERANK_TOP_N,
    )

    # Save results
    save_results(results_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
