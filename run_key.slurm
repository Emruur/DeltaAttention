#!/bin/bash
#SBATCH -J packing_delta35
#SBATCH -p gpu_h100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=0-24:00:00
#SBATCH --output=packing_delta35_%j.out
#SBATCH --error=packing_delta35_%j.err

source ~/.bashrc

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1

nvidia-smi
echo "CUDA version: $(nvcc --version | grep release)"

source /home/qchen/DeltaLLM/venv_delta/bin/activate

export SHARED_SCRATCH="/gpfs/home2/qchen/hf_cache"
mkdir -p $SHARED_SCRATCH
export HF_HOME=$SHARED_SCRATCH
export HUGGINGFACE_HUB_CACHE=$SHARED_SCRATCH
export TRANSFORMERS_CACHE=$SHARED_SCRATCH

source "$(dirname "${BASH_SOURCE[0]}")/.env.sh"

python compute_key_similarity.py
