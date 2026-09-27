"""
BM25 Retriever — lightweight keyword-based search over image captions and categories.

Uses the rank-bm25 library for sparse retrieval. This complements the dense
retriever: BM25 excels at exact keyword matches while dense retrieval handles
semantic similarity and paraphrasing.

Note: COCO captions are full natural-language sentences, so BM25 will
underperform dense retrieval on paraphrased/indirect queries. This contrast
is intentional and worth highlighting in the evaluation.
"""

import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METADATA_PATH, TOP_K


def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace + lowercase tokenizer.

    Strips punctuation and lowercases for BM25 matching.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return tokens


class BM25Retriever:
    """
    Keyword-based retriever using BM25 over COCO captions and categories.

    The index is built once from metadata.csv and reused for all queries.
    """

    def __init__(
        self,
        metadata_path: str = None,
        metadata_df: pd.DataFrame = None,
    ):
        """
        Build the BM25 index from metadata.

        Provide either a path to metadata.csv or a pre-loaded DataFrame.

        Args:
            metadata_path: Path to metadata.csv.
            metadata_df: Pre-loaded metadata DataFrame.
        """
        if metadata_df is not None:
            self.metadata = metadata_df.copy()
        elif metadata_path is not None:
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = pd.read_csv(str(METADATA_PATH))

        # Build searchable text corpus: combine caption + category for each image
        self.doc_ids = self.metadata["id"].tolist()
        self.corpus_texts = []  # Raw text for display
        self.corpus_tokens = []  # Tokenized for BM25

        for _, row in self.metadata.iterrows():
            caption = str(row.get("caption", ""))
            category = str(row.get("category", ""))
            combined = f"{caption} {category}"
            self.corpus_texts.append(combined)
            self.corpus_tokens.append(_tokenize(combined))

        # Build BM25 index
        self.bm25 = BM25Okapi(self.corpus_tokens)

        print(f"BM25 index built: {len(self.doc_ids)} documents")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Search captions and categories using BM25.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: id, score, image_path, caption, category
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")

        start_time = time.time()

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-K indices (descending score)
        top_indices = scores.argsort()[::-1][:top_k]

        search_time = time.time() - start_time

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue  # Skip zero-score results

            row = self.metadata.iloc[idx]
            results.append({
                "id": int(row["id"]),
                "score": score,
                "image_path": row["image_path"],
                "caption": row["caption"],
                "category": row.get("category", "unknown"),
            })

        # Attach timing
        if results:
            results[0]["_timing"] = {"bm25_ms": search_time * 1000}

        return results

    def get_scores_for_ids(
        self, query: str, doc_ids: List[int]
    ) -> Dict[int, float]:
        """
        Get BM25 scores for a specific subset of document IDs.

        Useful for hybrid retrieval where we need BM25 scores only
        for candidates retrieved by the dense retriever.

        Args:
            query: Search query.
            doc_ids: List of document IDs to score.

        Returns:
            Dict mapping doc_id → BM25 score.
        """
        query_tokens = _tokenize(query)
        all_scores = self.bm25.get_scores(query_tokens)

        # Build ID → index lookup
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        result = {}
        for doc_id in doc_ids:
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                result[doc_id] = float(all_scores[idx])
            else:
                result[doc_id] = 0.0

        return result
