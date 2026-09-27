"""
Reranker — re-scores top-N retrieval candidates using a cross-encoder model.

Architecture:
    5,000 documents → Dense/BM25 retrieval → Top 50 → Reranker → Top 10

Since pretrained rerankers are text-text cross-encoders, we use each
candidate's **caption** as the text proxy for the image. The reranker
scores (query_text, candidate_caption) pairs.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (lightweight, fast)
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RERANK_MODEL_NAME, RERANK_TOP_N


class Reranker:
    """
    Cross-encoder reranker that re-scores retrieval candidates.

    Uses the candidate's caption as a text proxy for the image,
    since cross-encoders operate on text pairs (query, passage).
    """

    def __init__(self, model_name: str = RERANK_MODEL_NAME, device: str = None):
        """
        Load the cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model name.
            device: "cuda" or "cpu". Auto-detects if None.
        """
        from sentence_transformers import CrossEncoder

        if device is None:
            from src.config import DEVICE
            device = DEVICE

        self.device = device
        print(f"Loading reranker: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)
        print("Reranker loaded.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Re-score and re-rank candidates using the cross-encoder.

        Each candidate must have a 'caption' field which is used as the
        text proxy for the image.

        Args:
            query: The search query text.
            candidates: List of candidate dicts from retrieval (must have 'caption').
            top_k: Number of top results to return after reranking.

        Returns:
            Re-ranked list of candidate dicts with updated scores.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty for reranking.")

        if not candidates:
            return []

        start_time = time.time()

        # Build (query, caption) pairs for the cross-encoder
        pairs = []
        for c in candidates:
            caption = c.get("caption", "")
            if not caption:
                caption = c.get("category", "unknown")
            pairs.append([query, caption])

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach reranker scores and sort
        reranked = []
        for i, candidate in enumerate(candidates):
            entry = candidate.copy()
            entry["rerank_score"] = float(scores[i])
            entry["original_score"] = candidate.get("score", 0.0)
            reranked.append(entry)

        # Sort by reranker score (descending)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Take top-k and set the main score to rerank_score
        results = reranked[:top_k]
        for r in results:
            r["score"] = r["rerank_score"]

        rerank_time = time.time() - start_time

        if results:
            results[0]["_timing"] = {
                "rerank_ms": rerank_time * 1000,
                "candidates_scored": len(candidates),
            }

        return results
