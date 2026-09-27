"""
Build the Qdrant vector index from pre-computed embeddings.

This script:
1. Loads saved embeddings (image_embeddings.npy, ids.npy)
2. Loads metadata.csv for payloads
3. Creates a Qdrant collection with cosine distance
4. Inserts all embeddings with metadata payloads
5. Verifies the index

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    IMAGE_EMBEDDINGS_PATH,
    IDS_PATH,
    METADATA_PATH,
    QDRANT_PATH,
    COLLECTION_NAME,
)
from src.ingestion.build_index import (
    get_qdrant_client,
    create_collection,
    insert_embeddings,
    verify_collection,
)


def main():
    print("=" * 60)
    print("Qdrant Index Builder")
    print("=" * 60)

    # Load embeddings
    print("\n[1/4] Loading embeddings ...")
    image_embeddings = np.load(str(IMAGE_EMBEDDINGS_PATH))
    ids = np.load(str(IDS_PATH))
    print(f"  Image embeddings: {image_embeddings.shape}")
    print(f"  IDs:              {ids.shape}")

    # Load metadata
    print("\n[2/4] Loading metadata ...")
    metadata_df = pd.read_csv(str(METADATA_PATH))
    print(f"  Metadata rows: {len(metadata_df)}")

    # Create Qdrant client and collection
    print(f"\n[3/4] Creating Qdrant collection ...")
    client = get_qdrant_client(QDRANT_PATH)
    vector_size = image_embeddings.shape[1]
    create_collection(client, COLLECTION_NAME, vector_size=vector_size, recreate=True)

    # Insert embeddings
    print(f"\n[4/4] Inserting embeddings ...")
    insert_embeddings(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=image_embeddings,
        ids=ids,
        metadata_df=metadata_df,
        batch_size=100,
    )

    # Verify
    verify_collection(client, COLLECTION_NAME, expected_count=len(ids))

    print(f"\n{'=' * 60}")
    print("Index built successfully!")
    print(f"  Qdrant path: {QDRANT_PATH}")
    print(f"  Collection:  {COLLECTION_NAME}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
