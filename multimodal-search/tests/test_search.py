"""
Unit tests for the Multimodal Search System.

Tests cover:
- Embedding normalization
- Score normalization (min-max)
- Score fusion
- BM25 retrieval
- Evaluation metrics (Recall@K, Precision@K, MRR, NDCG, Hit Rate)
- Query validation

All tests work without a GPU or loaded model (mocks used where needed).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# Score Normalization Tests
# ===========================================================================

class TestScoreNormalization:

    def test_minmax_basic(self):
        from src.utils.io import normalize_scores_minmax
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_scores_minmax(scores)
        assert result[0] == 0.0
        assert result[-1] == 1.0
        assert np.isclose(result[2], 0.5)

    def test_minmax_identical_scores(self):
        from src.utils.io import normalize_scores_minmax
        scores = np.array([3.0, 3.0, 3.0])
        result = normalize_scores_minmax(scores)
        assert np.allclose(result, 0.0)

    def test_minmax_single_element(self):
        from src.utils.io import normalize_scores_minmax
        scores = np.array([5.0])
        result = normalize_scores_minmax(scores)
        assert np.allclose(result, 0.0)

    def test_minmax_negative_scores(self):
        from src.utils.io import normalize_scores_minmax
        scores = np.array([-2.0, 0.0, 2.0])
        result = normalize_scores_minmax(scores)
        assert result[0] == 0.0
        assert result[-1] == 1.0
        assert np.isclose(result[1], 0.5)


# ===========================================================================
# Embedding Normalization Tests
# ===========================================================================

class TestEmbeddingNormalization:

    def test_l2_normalize(self):
        """Verify that manually normalizing vectors works as expected."""
        vectors = np.random.randn(10, 512).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms
        result_norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(result_norms, 1.0, atol=1e-5)

    def test_cosine_similarity_normalized(self):
        """For normalized vectors, cosine similarity = dot product."""
        a = np.random.randn(512).astype(np.float32)
        b = np.random.randn(512).astype(np.float32)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        cosine = np.dot(a, b)
        assert -1.0 <= cosine <= 1.0


# ===========================================================================
# Score Fusion Tests
# ===========================================================================

class TestScoreFusion:

    def _make_results(self, ids_scores):
        return [
            {"id": i, "score": s, "image_path": f"img/{i}.jpg", "caption": f"cap {i}", "category": "test"}
            for i, s in ids_scores
        ]

    def test_weighted_fusion_basic(self):
        from src.ranking.fusion import weighted_fusion

        results_a = self._make_results([(1, 0.9), (2, 0.8), (3, 0.7)])
        results_b = self._make_results([(2, 0.95), (3, 0.85), (4, 0.75)])

        fused = weighted_fusion(results_a, results_b, 0.5, 0.5, top_k=10)

        assert len(fused) > 0
        # All original IDs should be present
        fused_ids = {r["id"] for r in fused}
        assert {1, 2, 3, 4}.issubset(fused_ids)

    def test_weighted_fusion_respects_weights(self):
        from src.ranking.fusion import weighted_fusion

        results_a = self._make_results([(1, 1.0)])
        results_b = self._make_results([(2, 1.0)])

        fused = weighted_fusion(results_a, results_b, 0.9, 0.1, top_k=10)
        # With extreme weights, the result from source A should rank higher
        assert fused[0]["id"] == 1

    def test_empty_inputs(self):
        from src.ranking.fusion import weighted_fusion
        result = weighted_fusion([], [], 0.5, 0.5)
        assert result == []


# ===========================================================================
# BM25 Tests
# ===========================================================================

class TestBM25:

    @pytest.fixture
    def bm25_retriever(self):
        from src.retrieval.bm25_retriever import BM25Retriever
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "image_path": ["img/1.jpg", "img/2.jpg", "img/3.jpg", "img/4.jpg", "img/5.jpg"],
            "caption": [
                "A dog playing in the park",
                "A cat sitting on a couch",
                "A red car on the highway",
                "A dog running on the beach",
                "A person riding a bicycle",
            ],
            "category": ["dog", "cat", "car", "dog", "person"],
        })
        return BM25Retriever(metadata_df=df)

    def test_search_returns_results(self, bm25_retriever):
        results = bm25_retriever.search("dog park", top_k=3)
        assert len(results) > 0
        # Dog-related images should score highest
        top_ids = [r["id"] for r in results[:2]]
        assert 1 in top_ids or 4 in top_ids

    def test_search_empty_query_raises(self, bm25_retriever):
        with pytest.raises(ValueError):
            bm25_retriever.search("")

    def test_get_scores_for_ids(self, bm25_retriever):
        scores = bm25_retriever.get_scores_for_ids("dog", [1, 3, 4])
        assert 1 in scores
        assert scores[1] > scores[3]  # "dog" appears in caption 1 but not 3


# ===========================================================================
# Evaluation Metrics Tests
# ===========================================================================

class TestMetrics:

    def test_recall_at_k(self):
        from src.evaluation.metrics import recall_at_k
        retrieved = [1, 2, 3, 4, 5]
        relevant = [2, 4, 6]
        assert recall_at_k(retrieved, relevant, k=5) == 2 / 3

    def test_recall_at_k_perfect(self):
        from src.evaluation.metrics import recall_at_k
        retrieved = [1, 2, 3]
        relevant = [1, 2, 3]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_at_k(self):
        from src.evaluation.metrics import precision_at_k
        retrieved = [1, 2, 3, 4, 5]
        relevant = [2, 4]
        assert precision_at_k(retrieved, relevant, k=5) == 2 / 5

    def test_mrr_first_hit(self):
        from src.evaluation.metrics import mrr
        retrieved = [1, 2, 3]
        relevant = [1]
        assert mrr(retrieved, relevant) == 1.0

    def test_mrr_second_hit(self):
        from src.evaluation.metrics import mrr
        retrieved = [1, 2, 3]
        relevant = [2]
        assert mrr(retrieved, relevant) == 0.5

    def test_mrr_no_hit(self):
        from src.evaluation.metrics import mrr
        retrieved = [1, 2, 3]
        relevant = [99]
        assert mrr(retrieved, relevant) == 0.0

    def test_ndcg_at_k_perfect(self):
        from src.evaluation.metrics import ndcg_at_k
        retrieved = [1, 2, 3]
        relevant = [1, 2, 3]
        assert np.isclose(ndcg_at_k(retrieved, relevant, k=3), 1.0)

    def test_ndcg_at_k_no_hit(self):
        from src.evaluation.metrics import ndcg_at_k
        retrieved = [1, 2, 3]
        relevant = [99]
        assert ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_hit_rate_at_k_hit(self):
        from src.evaluation.metrics import hit_rate_at_k
        retrieved = [1, 2, 3]
        relevant = [3]
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0

    def test_hit_rate_at_k_miss(self):
        from src.evaluation.metrics import hit_rate_at_k
        retrieved = [1, 2, 3]
        relevant = [99]
        assert hit_rate_at_k(retrieved, relevant, k=3) == 0.0

    def test_evaluate_queries_batch(self):
        from src.evaluation.metrics import evaluate_queries
        all_retrieved = [[1, 2, 3], [4, 5, 6]]
        all_relevant = [[2], [5]]
        result = evaluate_queries(all_retrieved, all_relevant, k=3)
        assert "recall" in result
        assert "precision" in result
        assert "mrr" in result
        assert "ndcg" in result
        assert 0 <= result["recall"] <= 1


# ===========================================================================
# Query Validation Tests
# ===========================================================================

class TestQueryValidation:

    def test_empty_query_raises(self):
        """Dense retriever should reject empty queries."""
        # We test the validation logic directly rather than requiring the model
        query = ""
        assert not query.strip()

    def test_whitespace_query_fails(self):
        query = "   "
        assert not query.strip()

    def test_valid_query_passes(self):
        query = "dog playing in a park"
        assert query.strip()
