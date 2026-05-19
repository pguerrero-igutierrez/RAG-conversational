#!/bin/bash
#SBATCH --job-name=apps2-train-router
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/apps2/logs/train_router_%j.log
#SBATCH --error=/home/igutierrez134/apps2/logs/train_router_%j.err
#SBATCH --chdir=/home/igutierrez134/apps2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/apps2_3.11/bin/activate

export HF_TOKEN=HF_TOKEN
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

APPROACH=${APPROACH:-all}
SIZE=${SIZE:-}

echo "Router approach: $APPROACH"
if [ -n "$SIZE" ]; then
  echo "Router size: $SIZE per class"
  echo "Command: python -u scripts/router.py --train --approach $APPROACH --size $SIZE"
  python -u scripts/router.py --train --approach "$APPROACH" --size "$SIZE"
else
  echo "Router sizes: full learning curve"
  echo "Command: python -u scripts/router.py --train --approach $APPROACH"
  python -u scripts/router.py --train --approach "$APPROACH"
fi

echo "Job finished at $(date)"
