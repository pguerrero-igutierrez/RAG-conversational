#!/bin/bash
#SBATCH --job-name=apps2-load-dataset
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:0
#SBATCH --output=/home/igutierrez134/apps2/logs/load_dataset_%j.log
#SBATCH --error=/home/igutierrez134/apps2/logs/load_dataset_%j.err
#SBATCH --chdir=/home/igutierrez134/apps2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

source /home/igutierrez134/envs/apps2_3.11/bin/activate

export HF_TOKEN=HF_TOKEN
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"

python scripts/load_dataset.py

echo "Job finished at $(date)"
