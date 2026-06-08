#!/bin/bash -l
# Download the RAGTruth dataset into data/ragtruth/ (gitignored).
set -euo pipefail
base="https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
mkdir -p data/ragtruth
for f in source_info.jsonl response.jsonl; do
  echo "downloading $f ..."
  curl -fSL "$base/$f" -o "data/ragtruth/$f"
done
wc -l data/ragtruth/*.jsonl
