# Local Multimodal Search System

A local multimodal image search system that supports **text-to-image**, **image-to-image**, and **text + image-to-image** retrieval using multimodal embeddings, vector search, BM25, hybrid retrieval, score fusion, and reranking.

The project is designed as a **local research/portfolio project** for studying and demonstrating modern ML search architecture without unnecessary production infrastructure.

---

## 1. Project Overview

The system allows users to search a collection of images using:

* Text queries
* Image queries
* Text + image queries

The system combines:

* Multimodal embeddings
* Dense vector retrieval
* BM25 keyword retrieval
* Hybrid retrieval
* Multimodal score fusion
* Reranking
* Retrieval evaluation
* GPU inference
* Latency benchmarking

The entire system runs locally and does **not require cloud services**.

---

## 2. Supported Search Modes

### Text → Image

Enter a natural-language query and retrieve semantically relevant images.

Example:

```text
dog playing in a park
```

### Image → Image

Upload an image and retrieve visually/semantically similar images.

### Text + Image → Image

Combine an uploaded image with a text query.

Example:

```text
Image: uploaded scene

Text: a similar scene but at night
```

The text and image signals are combined using configurable score fusion.

---

## 3. Architecture

```text
                         User Query
                             │
              ┌──────────────┼──────────────┐
              │              │              │
           Text           Image        Text + Image
              │              │              │
              ▼              ▼              ▼
        Multimodal      Multimodal     Two Embeddings
         Encoder          Encoder            │
              │              │              │
              ▼              ▼              ▼
        Text Embedding   Image Embedding  Score Fusion
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                       Dense Retrieval
                             │
                             ▼
                           Qdrant
                             │
                             ▼
                       Candidate Images
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
         BM25 Search                    Dense Search
             │                               │
             └───────────────┬───────────────┘
                             ▼
                       Score Normalization
                             │
                             ▼
                         Hybrid Fusion
                             │
                             ▼
                          Reranker
                             │
                             ▼
                         Top-K Results
```

---

## 4. Technology Stack

| Component           | Technology                                 |
| ------------------- | ------------------------------------------ |
| Language            | Python 3.11+                               |
| Deep Learning       | PyTorch                                    |
| Numerical Computing | NumPy                                      |
| Data Processing     | Pandas                                     |
| Multimodal Model    | SigLIP / CLIP                              |
| Vector Database     | Qdrant                                     |
| Keyword Search      | BM25                                       |
| UI                  | Streamlit                                  |
| Optional API        | FastAPI                                    |
| GPU                 | CUDA                                       |
| Evaluation          | Recall@K, Precision@K, MRR, NDCG, Hit Rate |

The multimodal model interface is kept modular so that the selected model can be replaced later.

---

## 5. Dataset

### MS COCO val2017

The project uses **MS COCO val2017**.

The dataset contains:

* 5,000 images
* 5 human-written captions per image
* COCO object annotations
* 80 object categories

Only `val2017` is required.

### Dataset Files

```text
val2017.zip
annotations_trainval2017.zip
```

The annotation archive provides:

```text
captions_val2017.json
instances_val2017.json
```

### Metadata

The project generates:

```text
data/metadata.csv
```

Each row contains:

```text
id
image_path
caption
category
```

Example:

```text
id: 179765
image_path: data/images/000000179765.jpg
caption: A black Honda motorcycle parked in front of a garage.
category: motorcycle
```

The caption used in version 1 is the **first caption** associated with each image.

The category is derived from the object annotations. If no object instance is available:

```text
category = unknown
```

### Query Style

Queries should follow the natural scene descriptions used by COCO.

Examples:

```text
dog playing in a park
man riding a bicycle
people eating at a table
a plate of food on a table
children playing with a kite
```

---

## 6. Directory Structure

```text
multimodal-search/
│
├── data/
│   ├── images/
│   ├── metadata.csv
│   └── queries.json
│
├── embeddings/
│   ├── image_embeddings.npy
│   ├── text_embeddings.npy
│   └── ids.npy
│
├── src/
│   ├── config.py
│   │
│   ├── models/
│   │   └── multimodal_encoder.py
│   │
│   ├── ingestion/
│   │   └── build_index.py
│   │
│   ├── retrieval/
│   │   ├── dense_retriever.py
│   │   ├── bm25_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   ├── ranking/
│   │   ├── fusion.py
│   │   └── reranker.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       └── io.py
│
├── app/
│   └── streamlit_app.py
│
├── scripts/
│   ├── download_dataset.py
│   ├── generate_embeddings.py
│   ├── build_index.py
│   └── evaluate.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── tests/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 7. Hardware and Device Support

The system supports both GPU and CPU execution.

CUDA is automatically detected using:

```python
torch.cuda.is_available()
```

The device can be:

```text
cuda
```

or:

```text
cpu
```

Example environment configuration:

```text
DEVICE=cuda
```

or:

```text
DEVICE=cpu
```

CUDA device IDs should not be hard-coded.

---

## 8. Multimodal Encoder

The multimodal encoder is implemented in:

```text
src/models/multimodal_encoder.py
```

Main class:

```text
MultimodalEncoder
```

Responsibilities:

* Load the pretrained multimodal model
* Load the processor/tokenizer
* Select CPU/GPU
* Encode text
* Encode images
* Perform batch inference
* Normalize embeddings

Interface:

```python
encode_text(texts)
encode_images(images)
```

Both text and image embeddings are normalized before retrieval.

The model is loaded once and reused instead of being reloaded for every query.

---

## 9. Embedding Space

The selected model produces compatible text and image embeddings.

### Text

```text
"a dog playing in a park"
        │
        ▼
 Text Encoder
        │
        ▼
 Text Embedding
```

### Image

```text
Image
        │
        ▼
 Image Encoder
        │
        ▼
 Image Embedding
```

Embeddings are normalized and compared using cosine similarity.

For normalized vectors:

```text
cosine_similarity(a, b) = a · b
```

---

## 10. Offline Embedding Generation

Dataset embeddings are generated offline.

Script:

```text
scripts/generate_embeddings.py
```

Workflow:

```text
Metadata
   │
   ▼
Load Multimodal Model
   │
   ▼
Batch Image Processing
   │
   ▼
Image Embeddings
   │
   ├── Text/Caption Embeddings
   │
   ▼
Normalization
   │
   ▼
Save to Disk
```

Generated files:

```text
embeddings/image_embeddings.npy
embeddings/text_embeddings.npy
embeddings/ids.npy
```

The IDs are stored separately to ensure embeddings remain aligned with metadata.

---

## 11. Batch Processing

Images are processed in batches rather than individually.

```text
Images
   │
   ▼
Batch
   │
   ▼
GPU / CPU
   │
   ▼
Embeddings
```

Batch size is configurable:

```text
BATCH_SIZE=64
```

The implementation should reduce batch size when GPU memory is insufficient where practical.

---

## 12. Qdrant Vector Database

Qdrant is used as the local vector database.

Each vector contains:

```text
id
vector
payload
```

Example payload:

```json
{
  "image_id": 179765,
  "image_path": "data/images/000000179765.jpg",
  "caption": "A black Honda motorcycle parked in front of a garage.",
  "category": "motorcycle"
}
```

Cosine distance is used for vector similarity.

Index creation is handled by:

```text
scripts/build_index.py
```

The script:

1. Loads saved embeddings.
2. Creates the Qdrant collection.
3. Inserts image embeddings.
4. Stores metadata as payload.
5. Verifies the number of indexed vectors.

---

## 13. Dense Text-to-Image Search

Implementation:

```text
src/retrieval/dense_retriever.py
```

Function:

```python
search_text(query, top_k)
```

Pipeline:

```text
Text Query
    │
    ▼
Text Encoder
    │
    ▼
Query Embedding
    │
    ▼
Qdrant
    │
    ▼
Top-K Images
```

---

## 14. Image-to-Image Search

Function:

```python
search_image(image, top_k)
```

Pipeline:

```text
Uploaded Image
      │
      ▼
Image Encoder
      │
      ▼
Image Embedding
      │
      ▼
Qdrant
      │
      ▼
Similar Images
```

Results include:

* Retrieved image
* Similarity score
* Caption

---

## 15. Multimodal Search

Function:

```python
search_multimodal(image, text, top_k)
```

Two query embeddings are generated:

```text
image_query_embedding
text_query_embedding
```

Candidates are retrieved from both signals and combined.

```text
Image Search → Top 100
Text Search  → Top 100
       │
       ▼
Combine Candidate IDs
       │
       ▼
Calculate Image/Text Scores
       │
       ▼
Score Fusion
       │
       ▼
Final Ranking
```

---

## 16. Multimodal Score Fusion

Initial late-fusion formula:

```text
final_score =
    image_weight × image_score
    +
    text_weight × text_score
```

Default weights:

```text
image_weight = 0.6
text_weight  = 0.4
```

Weights are configurable.

Example experiments:

```text
0.8 / 0.2
0.6 / 0.4
0.5 / 0.5
0.3 / 0.7
```

---

## 17. Score Normalization

Scores from different retrieval systems should be normalized before fusion.

Reusable function:

```python
normalize_scores(scores)
```

The initial implementation uses:

```text
Min-Max Normalization
```

Raw scores from different retrieval systems should not be directly added.

---

## 18. BM25 Retrieval

A lightweight local BM25 implementation is used for keyword retrieval.

BM25 searches:

```text
caption
category
```

Example query:

```text
dog playing in a park
```

Output:

```text
document_id
bm25_score
```

No Elasticsearch or OpenSearch is required.

Because COCO captions are full natural-language sentences, BM25 is expected to be less effective for paraphrased or indirect queries. This provides a useful comparison with dense retrieval.

---

## 19. Hybrid Retrieval

Implementation:

```text
src/retrieval/hybrid_retriever.py
```

Pipeline:

```text
                Query
                  │
          ┌───────┴───────┐
          ▼               ▼
     Dense Search       BM25
          │               │
          └───────┬───────┘
                  ▼
          Score Normalization
                  │
                  ▼
             Score Fusion
                  │
                  ▼
                Top-K
```

Formula:

```text
hybrid_score =
    alpha × dense_score
    +
    beta × bm25_score
```

Constraint:

```text
alpha + beta = 1
```

Both parameters are configurable.

---

## 20. Reranking

The reranker is applied after initial retrieval.

Implementation:

```text
src/ranking/reranker.py
```

Pipeline:

```text
5,000 Images
     │
     ▼
Dense / BM25 Retrieval
     │
     ▼
Top 50
     │
     ▼
Reranker
     │
     ▼
Top 10
```

The reranker receives:

```text
query
candidate
```

and produces:

```text
relevance_score
```

A pretrained reranking model is used rather than training one from scratch.

For image candidates, the candidate's **caption** is used as the text representation for reranking.

The expensive reranker is not applied to the complete dataset.

---

## 21. Streamlit Application

The UI is implemented in:

```text
app/streamlit_app.py
```

The application provides three search modes:

1. Text Search
2. Image Search
3. Multimodal Search

The interface supports:

* Text input
* Image upload
* Top-K selection
* Search
* Retrieved images
* Similarity scores
* Captions/metadata
* Search latency

The model and search system are loaded once and cached appropriately.

Run the application with:

```bash
streamlit run app/streamlit_app.py
```

---

## 22. Evaluation Dataset

Evaluation queries are stored in:

```text
data/queries.json
```

Example:

```json
[
  {
    "query": "dog playing in a park",
    "relevant_ids": [4, 17, 82]
  },
  {
    "query": "person riding a bicycle",
    "relevant_ids": [23, 41, 90]
  },
  {
    "query": "a plate of food on a table",
    "relevant_ids": [11, 36, 58]
  }
]
```

The initial evaluation set contains approximately:

```text
50–100 queries
```

Queries should follow COCO's natural-scene caption style.

---

## 23. Evaluation Metrics

The system evaluates retrieval using:

### Recall@K

```text
Recall@K =
relevant retrieved documents
----------------------------
total relevant documents
```

### Precision@K

```text
Precision@K =
relevant retrieved documents
----------------------------
K
```

### MRR

Mean Reciprocal Rank.

### NDCG@K

Normalized Discounted Cumulative Gain.

### Hit Rate@K

Measures whether at least one relevant result appears within the top K results.

---

## 24. Retrieval Comparison

The evaluation compares:

1. BM25
2. Dense Retrieval
3. Hybrid Retrieval
4. Hybrid + Reranker

Example result format:

| Method            | Recall@10 | MRR@10 | NDCG@10 |
| ----------------- | --------: | -----: | ------: |
| BM25              |       ... |    ... |     ... |
| Dense             |       ... |    ... |     ... |
| Hybrid            |       ... |    ... |     ... |
| Hybrid + Reranker |       ... |    ... |     ... |

Results are saved to:

```text
results/evaluation.csv
```

---

## 25. Latency Benchmark

The system measures:

* Embedding latency
* Qdrant retrieval latency
* BM25 latency
* Fusion latency
* Reranking latency
* Total search latency

At least:

```text
100 queries
```

should be benchmarked.

Report:

```text
Mean
P50
P95
P99
```

Average latency alone is not sufficient.

---

## 26. GPU Benchmark

When CUDA is available, compare:

```text
CPU inference
        vs
GPU inference
```

Measure:

* Images/second
* Text queries/second
* Embedding latency
* GPU memory usage

GPU memory can be measured using PyTorch utilities.

Evaluation tests must also work without a GPU.

---

## 27. Experiments

The project includes the following experiments.

### Experiment 1 — Dense vs BM25

Compare semantic dense retrieval with keyword-based BM25 retrieval.

### Experiment 2 — Dense vs Hybrid

Measure the effect of combining dense and keyword retrieval.

### Experiment 3 — Hybrid vs Hybrid + Reranker

Measure the effect of reranking retrieved candidates.

### Experiment 4 — Multimodal Weights

Compare:

```text
Image 0.8 / Text 0.2
Image 0.6 / Text 0.4
Image 0.5 / Text 0.5
Image 0.3 / Text 0.7
```

### Experiment 5 — Retrieval Size

Compare:

```text
Top 10
Top 50
Top 100
Top 200
```

and analyze the latency/quality tradeoff.

---

## 28. Configuration

Configuration is maintained in:

```text
src/config.py
```

Supported configuration parameters include:

```text
MODEL_NAME
DEVICE
BATCH_SIZE
TOP_K

IMAGE_WEIGHT
TEXT_WEIGHT
BM25_WEIGHT

QDRANT_PATH
```

Environment variables can be configured using:

```text
.env
```

A template is provided as:

```text
.env.example
```

Do not commit `.env` to the repository.

---

## 29. Error Handling

The system handles:

* Missing images
* Corrupted images
* Invalid queries
* Empty queries
* Unavailable models
* Qdrant connection failures
* CUDA unavailability
* Insufficient GPU memory

Errors should provide useful messages to the user.

---

## 30. Testing

Tests cover:

```text
Embedding normalization
Score normalization
Score fusion
BM25 retrieval
Recall@K
Precision@K
MRR
NDCG
Query validation
```

Model inference should be mocked where possible for unit tests.

All unit tests should work without a GPU.

---

## 31. Installation

### Clone the repository

```bash
git clone <repository-url>
cd multimodal-search
```

### Create environment

```bash
python -m venv .venv
```

Activate it:

#### Linux/macOS

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 32. Development Workflow

The project is implemented in phases.

```text
Phase 1   Dataset
   ↓
Phase 2   Multimodal Model
   ↓
Phase 3   Offline Embeddings
   ↓
Phase 4   Qdrant Index
   ↓
Phase 5   Dense Text Search
   ↓
Phase 6   Image Search
   ↓
Phase 7   Multimodal Search
   ↓
Phase 8   BM25
   ↓
Phase 9   Hybrid Retrieval
   ↓
Phase 10  Reranking
   ↓
Phase 11  Evaluation
   ↓
Phase 12  Streamlit UI
   ↓
Phase 13  Benchmarking
```

Each phase should be:

1. Implemented
2. Executed
3. Tested
4. Debugged
5. Verified

before moving to the next phase.

---

## 33. Complete Workflow

### Step 1 — Download Dataset

Download:

```text
MS COCO val2017
captions_val2017.json
instances_val2017.json
```

Build:

```text
data/metadata.csv
```

### Step 2 — Generate Embeddings

```bash
python scripts/generate_embeddings.py
```

Produces:

```text
embeddings/image_embeddings.npy
embeddings/text_embeddings.npy
embeddings/ids.npy
```

### Step 3 — Build Qdrant Index

```bash
python scripts/build_index.py
```

### Step 4 — Start Streamlit

```bash
streamlit run app/streamlit_app.py
```

### Step 5 — Run Evaluation

```bash
python scripts/evaluate.py
```

### Step 6 — Run Benchmark

Measure:

```text
Latency
Throughput
GPU performance
Retrieval quality
```

---

## 34. Final User Experience

The application provides an interface similar to:

```text
╔══════════════════════════════════════════╗
║         MULTIMODAL SEARCH ENGINE         ║
╠══════════════════════════════════════════╣
║                                          ║
║ Search Mode:                             ║
║ (•) Text  ( ) Image  ( ) Multimodal     ║
║                                          ║
║ Query:                                   ║
║ [ dog playing in a park               ]  ║
║                                          ║
║ Upload Image:                            ║
║ [ Choose File ]                          ║
║                                          ║
║ Top K: [10]                              ║
║                                          ║
║              [ SEARCH ]                  ║
╠══════════════════════════════════════════╣
║ RESULTS                                  ║
║                                          ║
║ [IMAGE]    [IMAGE]    [IMAGE]            ║
║ Score .92  Score .89  Score .87          ║
║                                          ║
║ [IMAGE]    [IMAGE]    [IMAGE]            ║
║ Score .84  Score .82  Score .80          ║
║                                          ║
║ Search latency: 87 ms                    ║
╚══════════════════════════════════════════╝
```

---

## 35. Limitations

Current limitations include:

* Multimodal embedding quality depends on the selected pretrained model.
* The dataset contains only 5,000 images.
* Reranking introduces additional latency.
* GPU acceleration may be required for efficient large-scale inference.
* Weighted score fusion is manually configured rather than learned.
* BM25 depends on the available caption and category text.
* The evaluation dataset is relatively small.

---

## 36. Future Work

Possible extensions include:

* Better multimodal models
* Learned score fusion
* Personalized ranking
* Video search
* Larger datasets
* Fine-tuning
* ANN optimization
* Full COCO train2017
* Flickr30k integration

---

## 37. What Is Not Included

This project intentionally avoids unnecessary production infrastructure.

The following are **not required**:

```text
Kubernetes
Kafka
Redis
PostgreSQL
AWS
GCP
Azure
Prometheus
Grafana
Celery
Microservices
Load Balancers
Autoscaling
```

The objective is to understand and demonstrate the **ML search architecture**, not to build a production cloud platform.

---

## 38. Definition of Done

The project is complete when the following workflows work locally.

### Text Search

```text
COCO Dataset
     ↓
Generate Embeddings
     ↓
Build Qdrant Index
     ↓
Start Streamlit
     ↓
Enter Text Query
     ↓
Retrieve Images
```

### Image Search

```text
Upload Image
     ↓
Generate Image Embedding
     ↓
Qdrant Retrieval
     ↓
Retrieve Similar Images
```

### Multimodal Search

```text
Upload Image + Enter Text
          ↓
    Multimodal Fusion
          ↓
      Retrieve Results
```

### Evaluation

```text
Run Evaluation
      ↓
BM25 vs Dense vs Hybrid vs Reranker
      ↓
Recall / MRR / NDCG
```

### Benchmarking

```text
Run Benchmark
      ↓
Latency + Throughput + GPU Performance
```

The complete project remains **local** and does not require any cloud service.
