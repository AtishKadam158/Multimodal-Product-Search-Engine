"""
Configuration module for the Multimodal Search System.

All configurable parameters are centralized here.
Values can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load .env file if it exists
load_dotenv()

# =============================================================================
# Paths
# =============================================================================
# Project root is the parent of the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
METADATA_PATH = DATA_DIR / "metadata.csv"
QUERIES_PATH = DATA_DIR / "queries.json"

EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", PROJECT_ROOT / "embeddings"))
IMAGE_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "image_embeddings.npy"
TEXT_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "text_embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "ids.npy"

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))

# =============================================================================
# Device
# =============================================================================
_device_env = os.getenv("DEVICE", "auto").lower()

if _device_env == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif _device_env in ("cuda", "cpu"):
    DEVICE = _device_env
else:
    print(f"Warning: Unknown DEVICE '{_device_env}', falling back to auto-detect.")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Model
# =============================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

# =============================================================================
# Retrieval
# =============================================================================
TOP_K = int(os.getenv("TOP_K", "10"))

# Multimodal fusion weights (image query + text query)
IMAGE_WEIGHT = float(os.getenv("IMAGE_WEIGHT", "0.6"))
TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.4"))

# Hybrid retrieval weights (dense + BM25)
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# Reranker
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "50"))
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# =============================================================================
# Qdrant
# =============================================================================
QDRANT_PATH = os.getenv("QDRANT_PATH", str(PROJECT_ROOT / "qdrant_data"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multimodal_images")

# =============================================================================
# Dataset URLs (MS COCO val2017)
# =============================================================================
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("MULTIMODAL SEARCH SYSTEM — Configuration")
    print("=" * 60)
    print(f"  PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"  DEVICE:          {DEVICE}")
    print(f"  MODEL_NAME:      {MODEL_NAME}")
    print(f"  BATCH_SIZE:      {BATCH_SIZE}")
    print(f"  TOP_K:           {TOP_K}")
    print(f"  IMAGE_WEIGHT:    {IMAGE_WEIGHT}")
    print(f"  TEXT_WEIGHT:     {TEXT_WEIGHT}")
    print(f"  DENSE_WEIGHT:    {DENSE_WEIGHT}")
    print(f"  BM25_WEIGHT:     {BM25_WEIGHT}")
    print(f"  QDRANT_PATH:     {QDRANT_PATH}")
    print(f"  COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
    print("=" * 60)
