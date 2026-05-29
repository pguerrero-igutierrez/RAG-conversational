#!/bin/bash
#SBATCH --job-name=apps2-latency-nofb
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/apps2/logs/batch_latency_nofb_%j.log
#SBATCH --error=/home/igutierrez134/apps2/logs/batch_latency_nofb_%j.err
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

python scripts/main.py \
  --mode batch \
  --strategy always_rag \
  --no_feedback \
  --run_name latency_nofb_always_rag

python scripts/main.py \
  --mode batch \
  --strategy never_rag \
  --no_feedback \
  --run_name latency_nofb_never_rag

python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach frozen_lr \
  --router_size 15036 \
  --no_feedback \
  --run_name latency_nofb_router_rag_frozen_lr_15036

python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach setfit \
  --router_size 15036 \
  --no_feedback \
  --run_name latency_nofb_router_rag_setfit_15036

python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach finetune \
  --router_size 12500 \
  --no_feedback \
  --run_name latency_nofb_router_rag_finetune_12500

python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach setfit \
  --router_model_dir models/router_oracle/setfit/500_per_label \
  --no_feedback \
  --run_name latency_nofb_router_rag_oracle_setfit_500

python scripts/main.py \
  --mode batch \
  --strategy router_rag \
  --router_approach setfit \
  --router_model_dir models/router_oracle/setfit/64_per_label \
  --no_feedback \
  --run_name latency_nofb_router_rag_oracle_setfit_64

echo "Job finished at $(date)"
