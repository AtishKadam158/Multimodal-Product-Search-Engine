"""
Hybrid Retriever — combines dense (embedding) retrieval with BM25 keyword retrieval.

Pipeline:
    Query → Dense Search → candidates with dense scores
    Query → BM25 Search  → candidates with BM25 scores
         ↓
    Score Normalization (min-max)
         ↓
    Weighted Fusion: hybrid_score = alpha * dense + beta * bm25
         ↓
    Top-K results
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DENSE_WEIGHT, BM25_WEIGHT, TOP_K
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.ranking.fusion import hybrid_fusion


class HybridRetriever:
    """
    Combines dense vector retrieval with BM25 keyword retrieval for
    improved search quality.

    Dense retrieval handles semantic similarity and paraphrasing.
    BM25 handles exact keyword matches.
    The fusion balances both signals.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ):
        """
        Args:
            dense_retriever: Pre-initialized DenseRetriever.
            bm25_retriever: Pre-initialized BM25Retriever.
            dense_weight: Weight for dense scores (alpha). Default 0.7.
            bm25_weight: Weight for BM25 scores (beta). Default 0.3.
        """
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        dense_candidates: int = 100,
        bm25_candidates: int = 100,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid text search combining dense + BM25.

        Args:
            query: Natural language search query.
            top_k: Number of final results to return.
            dense_candidates: Number of candidates to retrieve from dense search.
            bm25_candidates: Number of candidates to retrieve from BM25.
            dense_weight: Override default dense weight for this query.
            bm25_weight: Override default BM25 weight for this query.

        Returns:
            Fused and re-ranked results.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")

        start_time = time.time()
        alpha = dense_weight if dense_weight is not None else self.dense_weight
        beta = bm25_weight if bm25_weight is not None else self.bm25_weight

        # Stage 1: Dense retrieval
        dense_start = time.time()
        dense_results = self.dense.search_text(query, top_k=dense_candidates)
        dense_time = time.time() - dense_start

        # Stage 2: BM25 retrieval
        bm25_start = time.time()
        bm25_results = self.bm25.search(query, top_k=bm25_candidates)
        bm25_time = time.time() - bm25_start

        # Stage 3: Fusion
        fusion_start = time.time()
        fused = hybrid_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            dense_weight=alpha,
            bm25_weight=beta,
            top_k=top_k,
        )
        fusion_time = time.time() - fusion_start

        total_time = time.time() - start_time

        # Attach timing
        if fused:
            fused[0]["_timing"] = {
                "dense_ms": dense_time * 1000,
                "bm25_ms": bm25_time * 1000,
                "fusion_ms": fusion_time * 1000,
                "total_ms": total_time * 1000,
            }

        return fused

    def search_with_image(
        self,
        query: str,
        image: Union[Image.Image, str, Path],
        top_k: int = TOP_K,
        dense_candidates: int = 100,
        bm25_candidates: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid multimodal search: text + image query with BM25 augmentation.

        Retrieves candidates using:
        1. Dense text search
        2. Dense image search
        3. BM25 text search

        Then fuses all results.

        Args:
            query: Text query.
            image: Query image.
            top_k: Number of final results.
            dense_candidates: Candidates per dense retrieval.
            bm25_candidates: Candidates from BM25.

        Returns:
            Fused results.
        """
        from src.ranking.fusion import multimodal_fusion

        start_time = time.time()

        # Dense text search
        text_results = self.dense.search_text(query, top_k=dense_candidates)

        # Dense image search
        image_results = self.dense.search_image(image, top_k=dense_candidates)

        # Multimodal fusion (text + image)
        mm_fused = multimodal_fusion(image_results, text_results, top_k=dense_candidates)

        # BM25 search
        bm25_results = self.bm25.search(query, top_k=bm25_candidates)

        # Hybrid fusion (multimodal dense + BM25)
        final = hybrid_fusion(
            dense_results=mm_fused,
            bm25_results=bm25_results,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
            top_k=top_k,
        )

        total_time = time.time() - start_time
        if final:
            final[0]["_timing"] = {"total_ms": total_time * 1000}

        return final
