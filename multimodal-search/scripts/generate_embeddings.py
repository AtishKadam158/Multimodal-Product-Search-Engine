"""
Generate embeddings for all images and captions in the dataset.

This script:
1. Loads metadata.csv
2. Loads the CLIP model once
3. Processes all images in batches → image_embeddings.npy
4. Processes all captions in batches → text_embeddings.npy
5. Saves corresponding IDs → ids.npy

Usage:
    python scripts/generate_embeddings.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    METADATA_PATH,
    EMBEDDINGS_DIR,
    IMAGE_EMBEDDINGS_PATH,
    TEXT_EMBEDDINGS_PATH,
    IDS_PATH,
    BATCH_SIZE,
    IMAGES_DIR,
)
from src.models.multimodal_encoder import MultimodalEncoder
from src.utils.io import ensure_dir


def main():
    print("=" * 60)
    print("Embedding Generation")
    print("=" * 60)

    # Load metadata
    df = pd.read_csv(str(METADATA_PATH))
    print(f"Loaded metadata: {len(df)} rows")

    # Initialize encoder (loads model once)
    encoder = MultimodalEncoder()

    ensure_dir(EMBEDDINGS_DIR)

    # ---- Generate image embeddings ----
    print(f"\n--- Image Embeddings ---")
    image_paths = []
    valid_ids = []

    for _, row in df.iterrows():
        # image_path in metadata is like "data/images/000000xxxxx.jpg"
        # We need the full path relative to project root
        img_path = PROJECT_ROOT / row["image_path"]
        if img_path.exists():
            image_paths.append(str(img_path))
            valid_ids.append(row["id"])
        else:
            print(f"  WARNING: Missing image: {img_path}")

    print(f"  Valid images: {len(image_paths)}")

    start_time = time.time()
    all_image_embeddings = []
    batch_size = encoder.batch_size

    for start in tqdm(range(0, len(image_paths), batch_size), desc="  Image batches"):
        batch = image_paths[start : start + batch_size]
        try:
            embs = encoder.encode_images(batch)
            all_image_embeddings.append(embs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  GPU OOM at batch_size={batch_size}. Reducing to {batch_size // 2}...")
                import torch
                torch.cuda.empty_cache()
                batch_size = max(batch_size // 2, 1)
                # Retry with smaller batch
                for sub_start in range(0, len(batch), batch_size):
                    sub_batch = batch[sub_start : sub_start + batch_size]
                    embs = encoder.encode_images(sub_batch)
                    all_image_embeddings.append(embs)
            else:
                raise

    image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    image_time = time.time() - start_time
    print(f"  Shape: {image_embeddings.shape}")
    print(f"  Time:  {image_time:.1f}s ({len(image_paths) / image_time:.1f} images/sec)")

    # ---- Generate text (caption) embeddings ----
    print(f"\n--- Text Embeddings ---")
    captions = df["caption"].tolist()
    # Ensure all captions are strings
    captions = [str(c) if pd.notna(c) else "" for c in captions]

    start_time = time.time()
    text_embeddings = encoder.encode_text(captions)
    text_time = time.time() - start_time
    print(f"  Shape: {text_embeddings.shape}")
    print(f"  Time:  {text_time:.1f}s ({len(captions) / text_time:.1f} texts/sec)")

    # ---- Save ----
    ids = np.array(valid_ids, dtype=np.int64)

    np.save(str(IMAGE_EMBEDDINGS_PATH), image_embeddings)
    np.save(str(TEXT_EMBEDDINGS_PATH), text_embeddings)
    np.save(str(IDS_PATH), ids)

    print(f"\n--- Saved ---")
    print(f"  {IMAGE_EMBEDDINGS_PATH} ({image_embeddings.shape})")
    print(f"  {TEXT_EMBEDDINGS_PATH}  ({text_embeddings.shape})")
    print(f"  {IDS_PATH}             ({ids.shape})")

    # Verify alignment
    assert len(image_embeddings) == len(ids), "Image embeddings / IDs mismatch!"
    assert len(text_embeddings) == len(df), "Text embeddings / metadata mismatch!"
    assert image_embeddings.shape[1] == text_embeddings.shape[1], "Embedding dim mismatch!"

    norms = np.linalg.norm(image_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "Image embeddings not normalized!"

    print(f"\n✓ All embeddings generated and verified.")
    print(f"  Embedding dim: {image_embeddings.shape[1]}")
    print(f"  Total time:    {image_time + text_time:.1f}s")


if __name__ == "__main__":
    main()
