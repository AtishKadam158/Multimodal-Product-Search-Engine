# Multimodal Search Engine

A local multimodal search system that retrieves semantically relevant images using text queries, image queries, or both — powered by CLIP embeddings, Qdrant vector search, BM25 keyword search, hybrid retrieval, and cross-encoder reranking.

## Architecture

```
Query (Text / Image / Both)
         ↓
   Multimodal Encoder (CLIP)
         ↓
    ┌─────┴─────┐
    ↓           ↓
Dense Search   BM25 Search
 (Qdrant)    (Keywords)
    ↓           ↓
    └─────┬─────┘
          ↓
   Score Normalization
          ↓
    Hybrid Fusion
          ↓
   Reranker (Cross-Encoder)
          ↓
      Top-K Results
```

## Technologies

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | OpenAI CLIP (ViT-B/32) | Produces aligned text + image embeddings in shared space |
| Vector DB | Qdrant (local) | Fast cosine similarity search, no Docker needed |
| Keyword Search | BM25 (rank-bm25) | Lightweight sparse retrieval, no Elasticsearch |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast cross-encoder for caption-based reranking |
| UI | Streamlit | Interactive search interface |
| Dataset | MS COCO val2017 | 5,000 images with natural-language captions |

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd multimodal-search

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The project uses **MS COCO val2017** — 5,000 images with 5 human-written captions each.

```bash
# Download images (~1 GB) + annotations and build metadata.csv
python scripts/download_dataset.py
```

This creates:
- `data/images/` — 5,000 JPEG images
- `data/metadata.csv` — id, image_path, caption, category

## Embedding Generation

```bash
# Generate CLIP embeddings for all images and captions
python scripts/generate_embeddings.py
```

Saves to `embeddings/`:
- `image_embeddings.npy` — (5000, 512) normalized image embeddings
- `text_embeddings.npy` — (5000, 512) normalized caption embeddings
- `ids.npy` — corresponding COCO image IDs

## Qdrant Indexing

```bash
# Build the vector index
python scripts/build_index.py
```

Creates a local Qdrant collection with cosine distance.

## Running the Search UI

```bash
# Start the Streamlit app
streamlit run app/streamlit_app.py
```

### Search Modes

1. **Text Search** — Enter a query like "dog playing in a park"
2. **Image Search** — Upload an image to find similar images
3. **Multimodal Search** — Combine text + image for refined results

### Retrieval Methods

- **Dense** — CLIP embedding similarity via Qdrant
- **BM25** — Keyword matching over captions
- **Hybrid** — Weighted fusion of Dense + BM25
- **Hybrid + Reranker** — Cross-encoder reranking of hybrid results

## Evaluation

```bash
# Compare all retrieval methods
python scripts/evaluate.py

# With custom K
python scripts/evaluate.py --k 20

# Skip reranker
python scripts/evaluate.py --no-reranker
```

Metrics computed: **Recall@K, Precision@K, MRR@K, NDCG@K, Hit Rate@K**

Results saved to `results/evaluation.csv`.

## Benchmarking

```bash
# Run latency + GPU benchmark
python scripts/benchmark.py --num-queries 100
```

Reports Mean, P50, P95, P99 for each pipeline stage.

## Running Tests

```bash
pytest tests/ -v
```

Tests cover normalization, fusion, BM25, all evaluation metrics, and query validation. All tests work without GPU.

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
DEVICE=auto          # auto, cuda, or cpu
MODEL_NAME=openai/clip-vit-base-patch32
BATCH_SIZE=64
TOP_K=10
IMAGE_WEIGHT=0.6     # For multimodal fusion
TEXT_WEIGHT=0.4
DENSE_WEIGHT=0.7     # For hybrid fusion
BM25_WEIGHT=0.3
```

## Project Structure

```
multimodal-search/
├── app/
│   └── streamlit_app.py          # Streamlit UI
├── data/
│   ├── images/                   # COCO val2017 images
│   ├── metadata.csv              # Image metadata
│   └── queries.json              # Evaluation queries
├── embeddings/                   # Pre-computed embeddings
├── scripts/
│   ├── download_dataset.py       # Download COCO dataset
│   ├── generate_embeddings.py    # Generate CLIP embeddings
│   ├── build_index.py            # Build Qdrant index
│   ├── evaluate.py               # Run evaluation
│   └── benchmark.py              # Latency benchmark
├── src/
│   ├── config.py                 # Centralized configuration
│   ├── models/
│   │   └── multimodal_encoder.py # CLIP encoder
│   ├── ingestion/
│   │   └── build_index.py        # Qdrant index management
│   ├── retrieval/
│   │   ├── dense_retriever.py    # Vector similarity search
│   │   ├── bm25_retriever.py     # Keyword search
│   │   └── hybrid_retriever.py   # Dense + BM25 fusion
│   ├── ranking/
│   │   ├── fusion.py             # Score fusion algorithms
│   │   └── reranker.py           # Cross-encoder reranker
│   ├── evaluation/
│   │   ├── metrics.py            # Recall, Precision, MRR, NDCG
│   │   └── evaluate.py           # Evaluation runner
│   └── utils/
│       └── io.py                 # Score normalization, I/O helpers
├── tests/
│   └── test_search.py            # Unit tests
├── results/                      # Evaluation & benchmark results
├── requirements.txt
├── .env.example
└── README.md
```

## Limitations

- **Dataset size**: 5,000 images is small relative to production search systems
- **Embedding quality**: CLIP ViT-B/32 is a base model; larger models (ViT-L/14) improve quality
- **Reranking**: Uses caption-based text proxy rather than true visual reranking
- **BM25 vs Dense**: BM25 underperforms on paraphrased queries since COCO captions are specific sentences
- **Fusion weights**: Fixed linear weights; learned fusion could improve results
- **GPU dependency**: Significantly faster with CUDA but works on CPU

## Future Work

- Larger/better multimodal models (SigLIP, CLIP ViT-L/14)
- Learned fusion weights
- Larger datasets (full COCO train2017, Flickr30k)
- Fine-tuning CLIP on domain-specific data
- Video search support
- Personalized ranking
- ANN index optimization (HNSW tuning)
