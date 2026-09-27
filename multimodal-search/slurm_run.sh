#!/bin/bash
#============================================================================
# SLURM Job Script — Multimodal Search Engine (V100 GPU)
#
# This script runs the full pipeline:
#   1. Setup environment
#   2. Download COCO val2017 dataset
#   3. Generate CLIP embeddings
#   4. Build Qdrant vector index
#   5. Run evaluation (BM25 vs Dense vs Hybrid vs Reranker)
#   6. Run latency benchmark
#
# Usage:
#   sbatch slurm_run.sh
#
# Estimated time: ~15 minutes on V100
#============================================================================

#SBATCH --job-name=multimodal-search
#SBATCH --output=slurm_logs/job_%j.out
#SBATCH --error=slurm_logs/job_%j.err
#SBATCH --partition=gpu                  # Change to your cluster's GPU partition
#SBATCH --gres=gpu:v100:1               # Request 1x V100 GPU
#SBATCH --mem=32G                        # 32 GB RAM (dataset + model + embeddings)
#SBATCH --cpus-per-task=4                # 4 CPU cores for data loading
#SBATCH --time=00:30:00                  # 30 min wall time (plenty of buffer)
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time:   $(date)"
echo "============================================"

# ------------------------------------------------------------------
# Step 0: Setup
# ------------------------------------------------------------------
# Navigate to project directory
# CHANGE THIS to your actual project path on the cluster
cd $HOME/multimodal-search

# Load modules (adjust to your cluster's module system)
# Uncomment and modify as needed:
# module load python/3.11
# module load cuda/11.8
# module load cudnn/8.6

# Create and activate virtual environment (first run only)
if [ ! -d "venv" ]; then
    echo "[Step 0] Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies (first run only)
echo "[Step 0] Installing dependencies..."
pip install --quiet -r requirements.txt

# Set environment variables
export DEVICE=cuda
export BATCH_SIZE=64
export CUDA_VISIBLE_DEVICES=0

# Create log directory
mkdir -p slurm_logs

# Verify GPU is available
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected! Running on CPU.')
"

# ------------------------------------------------------------------
# Step 1: Download Dataset (~2-5 min, network-bound)
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "[Step 1] Downloading COCO val2017 dataset..."
echo "============================================"
python3 scripts/download_dataset.py

# ------------------------------------------------------------------
# Step 2: Generate Embeddings (~2-3 min on V100)
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "[Step 2] Generating CLIP embeddings..."
echo "============================================"
python3 scripts/generate_embeddings.py

# ------------------------------------------------------------------
# Step 3: Build Qdrant Index (~15 sec)
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "[Step 3] Building Qdrant vector index..."
echo "============================================"
python3 scripts/build_index.py

# ------------------------------------------------------------------
# Step 4: Run Evaluation (~2 min)
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "[Step 4] Running evaluation..."
echo "============================================"
python3 scripts/evaluate.py --k 10

# ------------------------------------------------------------------
# Step 5: Run Benchmark (~3 min)
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "[Step 5] Running latency benchmark..."
echo "============================================"
python3 scripts/benchmark.py --num-queries 100

# ------------------------------------------------------------------
# Step 6: Run Tests
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "[Step 6] Running unit tests..."
echo "============================================"
python3 -m pytest tests/ -v --tb=short

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "ALL STEPS COMPLETE"
echo "End time: $(date)"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  results/evaluation.csv   — Retrieval method comparison"
echo "  results/benchmark.csv    — Latency measurements"
echo ""
echo "To run Streamlit UI (interactive job):"
echo "  srun --gres=gpu:v100:1 --mem=16G --time=01:00:00 --pty bash"
echo "  source venv/bin/activate"
echo "  streamlit run app/streamlit_app.py --server.port=8501"
echo "  # Then SSH tunnel: ssh -L 8501:$(hostname):8501 user@login-node"
