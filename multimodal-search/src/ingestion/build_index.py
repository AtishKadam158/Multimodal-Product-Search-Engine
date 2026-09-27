"""
Build and manage the Qdrant vector index for image embeddings.

This module handles:
- Creating/recreating the Qdrant collection
- Inserting image embeddings with metadata payloads
- Verifying index integrity
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDINGS_DIR,
    IMAGE_EMBEDDINGS_PATH,
    IDS_PATH,
    METADATA_PATH,
)


def get_qdrant_client(path: str = QDRANT_PATH) -> QdrantClient:
    """
    Create a Qdrant client using local file-based storage.

    No Docker or server required — Qdrant runs embedded.

    Args:
        path: Directory path for Qdrant's local storage.

    Returns:
        QdrantClient instance.
    """
    return QdrantClient(path=path)


def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    vector_size: int = 512,
    recreate: bool = False,
) -> None:
    """
    Create a Qdrant collection for storing image embeddings.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the collection.
        vector_size: Dimensionality of the embedding vectors.
        recreate: If True, delete and recreate the collection if it exists.
    """
    # Check if collection already exists
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        if recreate:
            print(f"  Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            print(f"  Collection '{collection_name}' already exists. Use recreate=True to overwrite.")
            return

    print(f"  Creating collection: {collection_name} (vector_size={vector_size}, distance=Cosine)")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    print("  Collection created.")


def insert_embeddings(
    client: QdrantClient,
    collection_name: str,
    embeddings: np.ndarray,
    ids: np.ndarray,
    metadata_df: pd.DataFrame,
    batch_size: int = 100,
) -> None:
    """
    Insert image embeddings into the Qdrant collection with metadata payloads.

    Each point contains:
        - id: integer point ID (using the COCO image_id)
        - vector: the image embedding
        - payload: {image_id, image_path, caption, category}

    Args:
        client: QdrantClient instance.
        collection_name: Target collection name.
        embeddings: np.ndarray of shape (N, dim).
        ids: np.ndarray of COCO image IDs, shape (N,).
        metadata_df: DataFrame with columns [id, image_path, caption, category].
        batch_size: Number of points to upsert per batch.
    """
    # Build a lookup from image_id to metadata row
    meta_lookup = metadata_df.set_index("id").to_dict("index")

    total = len(embeddings)
    inserted = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        batch_vectors = embeddings[start:end]

        points = []
        for i, (img_id, vector) in enumerate(zip(batch_ids, batch_vectors)):
            img_id_int = int(img_id)
            meta = meta_lookup.get(img_id_int, {})

            payload = {
                "image_id": img_id_int,
                "image_path": meta.get("image_path", ""),
                "caption": meta.get("caption", ""),
                "category": meta.get("category", "unknown"),
            }

            points.append(
                PointStruct(
                    id=img_id_int,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        inserted += len(points)

        if inserted % 500 == 0 or inserted == total:
            print(f"  Inserted {inserted}/{total} points")

    print(f"  Done. Total points inserted: {inserted}")


def verify_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Verify the Qdrant collection is properly built.

    Returns:
        Dictionary with collection info.
    """
    info = client.get_collection(collection_name)
    actual_count = info.points_count

    result = {
        "collection_name": collection_name,
        "points_count": actual_count,
        "vector_size": info.config.params.vectors.size,
        "distance": str(info.config.params.vectors.distance),
        "status": info.status,
    }

    print(f"\n  Collection: {collection_name}")
    print(f"  Points:     {actual_count}")
    print(f"  Vector dim: {result['vector_size']}")
    print(f"  Distance:   {result['distance']}")
    print(f"  Status:     {result['status']}")

    if expected_count is not None:
        if actual_count == expected_count:
            print(f"  ✓ Count matches expected ({expected_count})")
        else:
            print(f"  ✗ Count mismatch! Expected {expected_count}, got {actual_count}")

    return result
