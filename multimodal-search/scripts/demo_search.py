"""
Demo Search Script — runs search queries and prints/saves results.

No UI needed. Outputs results to console and saves to results/.

Usage:
    python scripts/demo_search.py
    python scripts/demo_search.py --query "dog playing in a park"
    python scripts/demo_search.py --query "person riding a bicycle" --method hybrid --top-k 20
    python scripts/demo_search.py --image data/images/000000179765.jpg
    python scripts/demo_search.py --query "at night" --image data/images/000000179765.jpg
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TOP_K, IMAGES_DIR, RESULTS_DIR


def print_results(results, title="Search Results"):
    """Print results in a formatted table to console."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Rank':<6} {'Score':<10} {'ID':<12} {'Category':<15} {'Caption'}")
    print(f"  {'-'*4:<6} {'-'*8:<10} {'-'*10:<12} {'-'*13:<15} {'-'*40}")

    for i, r in enumerate(results, 1):
        caption = r.get("caption", "")
        if len(caption) > 55:
            caption = caption[:52] + "..."
        print(f"  {i:<6} {r['score']:<10.4f} {r['id']:<12} {r.get('category', 'N/A'):<15} {caption}")

    print(f"{'=' * 80}\n")


def save_results_to_csv(results, output_path, query_info=""):
    """Save results to a CSV file."""
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "rank": i,
            "image_id": r["id"],
            "score": round(r["score"], 4),
            "image_path": r.get("image_path", ""),
            "caption": r.get("caption", ""),
            "category": r.get("category", ""),
        })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    return df


def copy_result_images(results, output_dir):
    """Copy result images to an output folder for easy viewing."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results, 1):
        src = PROJECT_ROOT / r.get("image_path", "")
        if src.exists():
            dst = out / f"rank_{i:02d}_id_{r['id']}_score_{r['score']:.3f}.jpg"
            shutil.copy2(src, dst)

    print(f"Result images copied to: {output_dir}")


def run_demo_queries(dense, bm25, hybrid, reranker, top_k):
    """Run a set of demo queries across all methods and print comparison."""
    demo_queries = [
        "dog playing in a park",
        "person riding a bicycle",
        "a plate of food on a table",
        "airplane flying in the sky",
        "cat sitting on a couch",
    ]

    print("\n" + "=" * 80)
    print("  DEMO: Comparing retrieval methods")
    print("=" * 80)

    for query in demo_queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  {'-' * 70}")

        # Dense
        start = time.time()
        dense_results = dense.search_text(query, top_k=top_k)
        dense_time = (time.time() - start) * 1000

        # BM25
        start = time.time()
        bm25_results = bm25.search(query, top_k=top_k)
        bm25_time = (time.time() - start) * 1000

        # Hybrid
        start = time.time()
        hybrid_results = hybrid.search(query, top_k=top_k)
        hybrid_time = (time.time() - start) * 1000

        # Print top-3 for each method
        print(f"  {'Method':<22} {'Top-1 ID':<12} {'Score':<10} {'Latency':<12} {'Top-1 Caption'}")
        print(f"  {'-'*20:<22} {'-'*10:<12} {'-'*8:<10} {'-'*10:<12} {'-'*30}")

        for name, res, lat in [
            ("Dense", dense_results, dense_time),
            ("BM25", bm25_results, bm25_time),
            ("Hybrid", hybrid_results, hybrid_time),
        ]:
            if res:
                cap = res[0].get("caption", "")[:40]
                print(f"  {name:<22} {res[0]['id']:<12} {res[0]['score']:<10.4f} {lat:<10.1f}ms {cap}")
            else:
                print(f"  {name:<22} {'No results':<12}")

        # Reranker
        if reranker:
            start = time.time()
            candidates = hybrid.search(query, top_k=50)
            reranked = reranker.rerank(query, candidates, top_k=top_k)
            rerank_time = (time.time() - start) * 1000
            if reranked:
                cap = reranked[0].get("caption", "")[:40]
                print(f"  {'Hybrid + Reranker':<22} {reranked[0]['id']:<12} {reranked[0]['score']:<10.4f} {rerank_time:<10.1f}ms {cap}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search — CLI Demo")
    parser.add_argument("--query", type=str, default=None, help="Text search query")
    parser.add_argument("--image", type=str, default=None, help="Path to query image")
    parser.add_argument("--method", type=str, default="dense",
                        choices=["dense", "bm25", "hybrid", "hybrid+reranker"],
                        help="Retrieval method")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of results")
    parser.add_argument("--save-images", action="store_true", help="Copy result images to results/")
    parser.add_argument("--demo", action="store_true", help="Run demo queries comparing all methods")
    args = parser.parse_args()

    print("=" * 80)
    print("  Multimodal Search Engine — CLI")
    print("=" * 80)

    # Load components
    print("\nLoading components...")
    from src.models.multimodal_encoder import MultimodalEncoder
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_retriever import HybridRetriever

    encoder = MultimodalEncoder()
    dense = DenseRetriever(encoder=encoder)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    reranker = None
    if args.method == "hybrid+reranker" or args.demo:
        try:
            from src.ranking.reranker import Reranker
            reranker = Reranker()
        except Exception as e:
            print(f"Warning: Reranker unavailable: {e}")

    # Demo mode
    if args.demo:
        run_demo_queries(dense, bm25, hybrid, reranker, args.top_k)
        return

    # Single query mode
    if args.query is None and args.image is None:
        print("No query provided. Use --query, --image, or --demo")
        print("Examples:")
        print('  python scripts/demo_search.py --query "dog playing in a park"')
        print('  python scripts/demo_search.py --image data/images/000000179765.jpg')
        print('  python scripts/demo_search.py --demo')
        return

    start_time = time.time()

    if args.query and args.image:
        # Multimodal search
        from PIL import Image
        from src.ranking.fusion import multimodal_fusion

        print(f"\nMultimodal search: text=\"{args.query}\" + image=\"{args.image}\"")
        image = Image.open(args.image).convert("RGB")
        text_results = dense.search_text(args.query, top_k=100)
        image_results = dense.search_image(image, top_k=100)
        results = multimodal_fusion(image_results, text_results, top_k=args.top_k)
        title = f"Multimodal: \"{args.query}\" + {Path(args.image).name}"

    elif args.query:
        # Text search
        print(f"\nText search: \"{args.query}\" (method={args.method})")
        if args.method == "dense":
            results = dense.search_text(args.query, top_k=args.top_k)
        elif args.method == "bm25":
            results = bm25.search(args.query, top_k=args.top_k)
        elif args.method == "hybrid":
            results = hybrid.search(args.query, top_k=args.top_k)
        elif args.method == "hybrid+reranker":
            candidates = hybrid.search(args.query, top_k=50)
            results = reranker.rerank(args.query, candidates, top_k=args.top_k)
        title = f"Text [{args.method}]: \"{args.query}\""

    elif args.image:
        # Image search
        from PIL import Image
        print(f"\nImage search: {args.image}")
        image = Image.open(args.image).convert("RGB")
        results = dense.search_image(image, top_k=args.top_k)
        title = f"Image: {Path(args.image).name}"

    total_time = (time.time() - start_time) * 1000

    # Print results
    print_results(results, title)
    print(f"  Total latency: {total_time:.0f} ms")
    print(f"  Results: {len(results)}")

    # Save
    out_dir = Path(RESULTS_DIR)
    save_results_to_csv(results, out_dir / "search_results.csv")

    if args.save_images:
        copy_result_images(results, out_dir / "search_images")


if __name__ == "__main__":
    main()
