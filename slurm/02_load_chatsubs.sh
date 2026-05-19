#!/bin/bash
#SBATCH --job-name=apps2-load-chatsubs
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:0
#SBATCH --output=/home/igutierrez134/apps2/logs/load_chatsubs_%j.log
#SBATCH --error=/home/igutierrez134/apps2/logs/load_chatsubs_%j.err
#SBATCH --chdir=/home/igutierrez134/apps2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/apps2_3.11/bin/activate

export HF_TOKEN=HF_TOKEN
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"

mkdir -p data/raw
if [ ! -f data/raw/ChatSubs.tar.gz ]; then
  echo "Downloading ChatSubs archive..."
  wget -O data/raw/ChatSubs.tar.gz https://zenodo.org/records/8220853/files/ChatSubs.tar.gz
else
  echo "Found existing ChatSubs archive: data/raw/ChatSubs.tar.gz"
fi

python -u scripts/load_chatsubs.py
python -u scripts/load_microsoft_chitchat.py

echo "Job finished at $(date)"
