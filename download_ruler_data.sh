#!/bin/bash
set -e

export HF_TOKEN="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"
export HF_DATASETS_TRUST_REMOTE_CODE=1

JSON_DIR="$(dirname "$0")/RULER/scripts/data/synthetic/json"
cd "$JSON_DIR"

echo "[1/2] Downloading Paul Graham essays..."
python download_paulgraham_essay.py

echo "[2/2] Downloading SQuAD and HotpotQA..."
bash download_qa_dataset.sh

echo "Done. Data saved to: $JSON_DIR"
