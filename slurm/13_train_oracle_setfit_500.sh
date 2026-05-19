#!/bin/bash
#SBATCH --job-name=apps2-oracle-setfit-500
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/apps2/logs/train_oracle_setfit_500_%j.log
#SBATCH --error=/home/igutierrez134/apps2/logs/train_oracle_setfit_500_%j.err
#SBATCH --chdir=/home/igutierrez134/apps2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

source /home/igutierrez134/envs/apps2_3.11/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/train_oracle_setfit.py \
  --data corpus/oracle_train_sqac_500_per_label.jsonl \
  --save_dir models/router_oracle/setfit/500_per_label \
  --num_epochs 4 \
  --num_iterations 24 \
  --batch_size 16

echo "Job finished at $(date)"
