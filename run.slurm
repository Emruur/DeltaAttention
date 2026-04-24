#!/bin/bash
#SBATCH -J v4
#SBATCH -p gpu_h100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=0-08:30:00
#SBATCH --output=setup_%j.out
#SBATCH --error=setup_%j.err

source ~/.bashrc

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1

nvidia-smi
echo "CUDA version: $(nvcc --version | grep release)"

source /home/qchen/DeltaLLM/venv_delta/bin/activate

export SHARED_SCRATCH="/gpfs/home2/qchen/hf_cache"
mkdir -p $SHARED_SCRATCH

# 2. Force Hugging Face to use that folder for EVERYTHING
export HF_HOME=$SHARED_SCRATCH
export HUGGINGFACE_HUB_CACHE=$SHARED_SCRATCH
export TRANSFORMERS_CACHE=$SHARED_SCRATCH

export HF_TOKEN="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"
srun --export=ALL,HF_TOKEN python eval_all.py --llama --experiment_type row_delta --long_bench --exp_name experiment_row_delta_1
