"""
Dense Retriever — performs semantic search using vector similarity.

Supports three search modes:
    1. Text → Image search  (encode query text, search Qdrant)
    2. Image → Image search (encode query image, search Qdrant)
    3. Multimodal search     (combine text + image embeddings, search Qdrant)
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import QDRANT_PATH, COLLECTION_NAME, TOP_K
from src.models.multimodal_encoder import MultimodalEncoder


class DenseRetriever:
    """
    Retrieves semantically similar images from Qdrant using dense embeddings.

    The encoder is loaded once and reused across all queries.
    """

    def __init__(
        self,
        encoder: Optional[MultimodalEncoder] = None,
        qdrant_path: str = QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        Args:
            encoder: Pre-loaded MultimodalEncoder. If None, creates a new one.
            qdrant_path: Path to local Qdrant storage.
            collection_name: Qdrant collection to search.
        """
        self.encoder = encoder or MultimodalEncoder()
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

    def search_text(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Text → Image search.

        Pipeline: query text → text encoder → embedding → Qdrant → top-K images

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: id, score, image_path, caption, category
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")

        start_time = time.time()

        # Encode query
        query_embedding = self.encoder.encode_text_single(query)
        encode_time = time.time() - start_time

        # Search Qdrant
        search_start = time.time()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )
        search_time = time.time() - search_start

        total_time = time.time() - start_time

        return self._format_results(results, {
            "encode_ms": encode_time * 1000,
            "search_ms": search_time * 1000,
            "total_ms": total_time * 1000,
        })

    def search_image(
        self,
        image: Union[Image.Image, str, Path],
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Image → Image search.

        Pipeline: query image → image encoder → embedding → Qdrant → similar images

        Args:
            image: Query image (PIL Image or file path).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: id, score, image_path, caption, category
        """
        start_time = time.time()

        # Encode query image
        query_embedding = self.encoder.encode_image_single(image)
        encode_time = time.time() - start_time

        # Search Qdrant
        search_start = time.time()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )
        search_time = time.time() - search_start

        total_time = time.time() - start_time

        return self._format_results(results, {
            "encode_ms": encode_time * 1000,
            "search_ms": search_time * 1000,
            "total_ms": total_time * 1000,
        })

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Search Qdrant using a pre-computed embedding vector.

        Useful for multimodal fusion where the combined embedding
        is computed externally.

        Args:
            embedding: 1-D normalized embedding vector.
            top_k: Number of results.

        Returns:
            List of result dicts.
        """
        start_time = time.time()

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=top_k,
        )

        search_time = time.time() - start_time

        return self._format_results(results, {
            "search_ms": search_time * 1000,
            "total_ms": search_time * 1000,
        })

    def get_embeddings_for_query(
        self, query: str = None, image: Union[Image.Image, str, Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a text query and/or image without searching.

        Returns:
            Dict with optional keys 'text_embedding' and 'image_embedding'.
        """
        result = {}
        if query:
            result["text_embedding"] = self.encoder.encode_text_single(query)
        if image is not None:
            result["image_embedding"] = self.encoder.encode_image_single(image)
        return result

    @staticmethod
    def _format_results(
        qdrant_results, timing: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Convert Qdrant search results to a standardized list of dicts."""
        formatted = []
        for hit in qdrant_results:
            formatted.append({
                "id": hit.id,
                "score": hit.score,
                "image_path": hit.payload.get("image_path", ""),
                "caption": hit.payload.get("caption", ""),
                "category": hit.payload.get("category", ""),
            })

        # Attach timing info to the list as an attribute (accessible but non-intrusive)
        if formatted:
            formatted[0]["_timing"] = timing

        return formatted
