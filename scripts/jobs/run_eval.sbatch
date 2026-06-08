#!/bin/bash -l
#SBATCH -J sumlens
#SBATCH -A tegawende.bissyande
#SBATCH -p l40s-trux
#SBATCH --qos=iris-snt-trux-long
#SBATCH -N 1 -c 8 --mem 24G
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH -o /mnt/scratch/users/betteib/a-s/logs/sumlens_%j.out
#SBATCH -e /mnt/scratch/users/betteib/a-s/logs/sumlens_%j.err

# Full RAGTruth evaluation: signals A (LettuceDetect) + B (NLI) over the summary
# split -> fit fusion LR -> ablation table. Attribution (C) is off (RAGTruth
# summaries are not from our model). Each step writes its own file, so progress
# survives between steps.

set -e
export MODULEPATH=/opt/apps/easybuild/environment/modules:/cvmfs/software.eessi.io/init/modules:/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/hopper/modules/all:/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/saphirerapids/modules/all:/opt/apps/easybuild/systems/binary/rhel810-20250803/2023b/generic/modules/all

cd /mnt/scratch/users/betteib/sum_lens
export HF_HOME=/mnt/scratch/users/betteib/hf
export TRANSFORMERS_CACHE=/mnt/scratch/users/betteib/hf
# ModernBERT (LettuceDetect) defaults to torch.compile on GPU, which JIT-compiles
# a CUDA util via gcc and needs Python.h (python-devel) -- absent on compute nodes.
# Disable dynamo so torch.compile is a no-op; inference correctness is unchanged.
export TORCHDYNAMO_DISABLE=1
mkdir -p /mnt/scratch/users/betteib/a-s/logs

echo "=== sumlens eval | job $SLURM_JOB_ID | $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no gpu visible"

# --- GPU env: CUDA torch (default wheel on the GPU node) + signal deps only ---
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -q --upgrade pip
# node driver 565.57.01 supports up to CUDA 12.7; the default PyPI wheel is cu130
# (needs driver >=580) and silently falls back to CPU. Pin the cu126 wheel.
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q "transformers<5" lettucedetect nltk pdfplumber scikit-learn "pydantic>=2"
pip install -q -e . --no-deps
python -m nltk.downloader punkt punkt_tab
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# --- pipeline (each step skipped if its output already exists, so a resubmit
# after a crash/timeout resumes instead of redoing finished work) ---
echo ">>> extract features (train)"; [ -f features_train.csv ] || python scripts/extract_features.py --split train --data-dir data/ragtruth --out features_train.csv
echo ">>> extract features (test)";  [ -f features_test.csv ]  || python scripts/extract_features.py --split test  --data-dir data/ragtruth --out features_test.csv
echo ">>> train fusion";             [ -f models/fusion.pkl ]  || python scripts/train_fusion.py --features features_train.csv --out-dir models
echo ">>> ablation table";           [ -f ablation.csv ]       || python scripts/evaluate.py --train features_train.csv --test features_test.csv --out ablation.csv

echo "=== DONE $(date) ==="
cat ablation.csv
