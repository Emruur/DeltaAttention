#!/bin/bash
#SBATCH -J ruler_delta
#SBATCH -p gpu_h100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=0-48:00:00
#SBATCH --output=ruler_%j.out
#SBATCH --error=ruler_%j.err

source ~/.bashrc

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1

nvidia-smi
echo "CUDA version: $(nvcc --version | grep release)"

source /home/qchen/DeltaLLM/venv_delta/bin/activate

pip install -q wonderwords tenacity

export HF_HOME="/gpfs/home2/qchen/hf_cache"
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_TOKEN="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"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# RULER scripts must be run from RULER/scripts/
cd RULER/scripts

for MODEL in delta-baseline delta-row-t15 delta-row-t17; do
    echo ""
    echo "========================================"
    echo "  Running RULER for model: $MODEL"
    echo "========================================"
    bash run.sh $MODEL synthetic
done
