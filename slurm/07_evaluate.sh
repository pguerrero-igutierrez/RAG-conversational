#!/bin/bash
#SBATCH --job-name=apps2-evaluate
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/apps2/logs/evaluate_%j.log
#SBATCH --error=/home/igutierrez134/apps2/logs/evaluate_%j.err
#SBATCH --chdir=/home/igutierrez134/apps2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

source /home/igutierrez134/envs/apps2_3.11/bin/activate

export HF_TOKEN="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"
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/evaluate.py

echo "Job finished at $(date)"
