"""
Latency and GPU Benchmark script.

Measures:
- Embedding latency (text + image)
- Qdrant retrieval latency
- BM25 latency
- Fusion latency
- Reranking latency
- Total search latency

Reports: Mean, P50 (median), P95, P99

If CUDA is available, compares CPU vs GPU inference.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --num-queries 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, IMAGES_DIR, METADATA_PATH


def percentile_stats(latencies: list) -> dict:
    """Calculate Mean, P50, P95, P99 from latency measurements (in ms)."""
    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def benchmark_embedding(encoder, num_queries=100):
    """Benchmark text and image embedding latency."""
    print("\n--- Embedding Latency ---")

    # Text encoding
    texts = [f"a sample query about scene number {i}" for i in range(num_queries)]
    text_latencies = []
    for t in texts:
        start = time.perf_counter()
        encoder.encode_text_single(t)
        text_latencies.append((time.perf_counter() - start) * 1000)

    text_stats = percentile_stats(text_latencies)
    print(f"  Text encoding ({num_queries} queries):")
    print(f"    Mean: {text_stats['mean_ms']:.1f}ms | P50: {text_stats['p50_ms']:.1f}ms | "
          f"P95: {text_stats['p95_ms']:.1f}ms | P99: {text_stats['p99_ms']:.1f}ms")

    # Image encoding (use a dummy image)
    dummy_img = Image.new("RGB", (224, 224), (128, 100, 80))
    img_latencies = []
    for _ in range(min(num_queries, 50)):  # Fewer image tests (slower)
        start = time.perf_counter()
        encoder.encode_image_single(dummy_img)
        img_latencies.append((time.perf_counter() - start) * 1000)

    img_stats = percentile_stats(img_latencies)
    print(f"  Image encoding ({len(img_latencies)} queries):")
    print(f"    Mean: {img_stats['mean_ms']:.1f}ms | P50: {img_stats['p50_ms']:.1f}ms | "
          f"P95: {img_stats['p95_ms']:.1f}ms | P99: {img_stats['p99_ms']:.1f}ms")

    return {"text_encoding": text_stats, "image_encoding": img_stats}


def benchmark_retrieval(dense_retriever, bm25_retriever, num_queries=100):
    """Benchmark retrieval latency for dense and BM25."""
    print("\n--- Retrieval Latency ---")
    queries = [f"scene description query {i}" for i in range(num_queries)]

    # Dense
    dense_latencies = []
    for q in queries:
        start = time.perf_counter()
        dense_retriever.search_text(q, top_k=10)
        dense_latencies.append((time.perf_counter() - start) * 1000)

    dense_stats = percentile_stats(dense_latencies)
    print(f"  Dense retrieval ({num_queries} queries):")
    print(f"    Mean: {dense_stats['mean_ms']:.1f}ms | P50: {dense_stats['p50_ms']:.1f}ms | "
          f"P95: {dense_stats['p95_ms']:.1f}ms | P99: {dense_stats['p99_ms']:.1f}ms")

    # BM25
    bm25_latencies = []
    for q in queries:
        start = time.perf_counter()
        bm25_retriever.search(q, top_k=10)
        bm25_latencies.append((time.perf_counter() - start) * 1000)

    bm25_stats = percentile_stats(bm25_latencies)
    print(f"  BM25 retrieval ({num_queries} queries):")
    print(f"    Mean: {bm25_stats['mean_ms']:.1f}ms | P50: {bm25_stats['p50_ms']:.1f}ms | "
          f"P95: {bm25_stats['p95_ms']:.1f}ms | P99: {bm25_stats['p99_ms']:.1f}ms")

    return {"dense_retrieval": dense_stats, "bm25_retrieval": bm25_stats}


def benchmark_hybrid(hybrid_retriever, num_queries=100):
    """Benchmark hybrid retrieval (dense + BM25 + fusion)."""
    print("\n--- Hybrid Retrieval Latency ---")
    queries = [f"scene description query {i}" for i in range(num_queries)]

    latencies = []
    for q in queries:
        start = time.perf_counter()
        hybrid_retriever.search(q, top_k=10)
        latencies.append((time.perf_counter() - start) * 1000)

    stats = percentile_stats(latencies)
    print(f"  Hybrid ({num_queries} queries):")
    print(f"    Mean: {stats['mean_ms']:.1f}ms | P50: {stats['p50_ms']:.1f}ms | "
          f"P95: {stats['p95_ms']:.1f}ms | P99: {stats['p99_ms']:.1f}ms")

    return {"hybrid_retrieval": stats}


def benchmark_reranker(reranker, hybrid_retriever, num_queries=50):
    """Benchmark reranking latency."""
    print("\n--- Reranking Latency ---")
    queries = [f"scene description query {i}" for i in range(num_queries)]

    latencies = []
    for q in queries:
        candidates = hybrid_retriever.search(q, top_k=50)
        start = time.perf_counter()
        reranker.rerank(q, candidates, top_k=10)
        latencies.append((time.perf_counter() - start) * 1000)

    stats = percentile_stats(latencies)
    print(f"  Reranking ({num_queries} queries, top-50 → top-10):")
    print(f"    Mean: {stats['mean_ms']:.1f}ms | P50: {stats['p50_ms']:.1f}ms | "
          f"P95: {stats['p95_ms']:.1f}ms | P99: {stats['p99_ms']:.1f}ms")

    return {"reranking": stats}


def benchmark_gpu(encoder, num_images=200):
    """Compare CPU vs GPU inference if CUDA is available."""
    import torch

    if not torch.cuda.is_available():
        print("\n--- GPU Benchmark: SKIPPED (no CUDA) ---")
        return {}

    print("\n--- GPU Benchmark ---")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    dummy_images = [Image.new("RGB", (224, 224), (i % 256, 100, 80)) for i in range(num_images)]

    # GPU batch throughput
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    encoder.encode_images(dummy_images)
    gpu_time = time.perf_counter() - start
    gpu_memory = torch.cuda.max_memory_allocated() / 1e6

    gpu_throughput = num_images / gpu_time
    print(f"  GPU: {gpu_throughput:.1f} images/sec | {gpu_time:.2f}s total | {gpu_memory:.0f} MB peak")

    # Text throughput
    texts = [f"sample query number {i}" for i in range(num_images)]
    start = time.perf_counter()
    encoder.encode_text(texts)
    text_time = time.perf_counter() - start
    text_throughput = num_images / text_time
    print(f"  GPU text: {text_throughput:.1f} texts/sec | {text_time:.2f}s total")

    return {
        "gpu_image_throughput": gpu_throughput,
        "gpu_text_throughput": text_throughput,
        "gpu_memory_mb": gpu_memory,
    }


def main():
    parser = argparse.ArgumentParser(description="Latency & GPU Benchmark")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--no-reranker", action="store_true", help="Skip reranker benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("Multimodal Search — Benchmark")
    print("=" * 60)

    from src.models.multimodal_encoder import MultimodalEncoder
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_retriever import HybridRetriever

    encoder = MultimodalEncoder()
    dense = DenseRetriever(encoder=encoder)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    all_results = {}

    all_results.update(benchmark_embedding(encoder, args.num_queries))
    all_results.update(benchmark_retrieval(dense, bm25, args.num_queries))
    all_results.update(benchmark_hybrid(hybrid, args.num_queries))

    if not args.no_reranker:
        try:
            from src.ranking.reranker import Reranker
            reranker = Reranker()
            all_results.update(benchmark_reranker(reranker, hybrid, min(args.num_queries, 50)))
        except Exception as e:
            print(f"\nReranker benchmark skipped: {e}")

    all_results.update(benchmark_gpu(encoder))

    # Save results
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for component, stats in all_results.items():
        if isinstance(stats, dict) and "mean_ms" in stats:
            rows.append({"component": component, **stats})

    if rows:
        df = pd.DataFrame(rows)
        out_path = out_dir / "benchmark.csv"
        df.to_csv(out_path, index=False)
        print(f"\n\nBenchmark results saved to: {out_path}")
        print(df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
