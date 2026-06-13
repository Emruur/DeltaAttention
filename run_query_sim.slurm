#!/bin/bash
#SBATCH -J query_sim
#SBATCH -p gpu_h100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=0-02:00:00
#SBATCH --output=query_sim_%j.out
#SBATCH --error=query_sim_%j.err

source ~/.bashrc

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1

nvidia-smi

source /scratch-shared/qchen/venv_delta/bin/activate

export SHARED_SCRATCH="/gpfs/home2/qchen/hf_cache"
mkdir -p $SHARED_SCRATCH
export HF_HOME=$SHARED_SCRATCH
export HUGGINGFACE_HUB_CACHE=$SHARED_SCRATCH
export TRANSFORMERS_CACHE=$SHARED_SCRATCH

ENV_SH="$(dirname "${BASH_SOURCE[0]}")/.env.sh"
[ -f "$ENV_SH" ] && source "$ENV_SH"

python visualize_query_similarity.py
