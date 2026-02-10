#!/bin/bash
# #SBATCH -p xudong-gpu
# #SBATCH -A xudong-lab
#SBATCH -p gpu
#SBATCH --mem 64G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1
#SBATCH --time 02-00:00:00 #Time for the job to run
#SBATCH --job-name ptm_training
#SBATCH --mail-type begin,end,fail,requeue 

module load miniconda3
#module load cuda/11.7.0_gcc_9.5.0

# Activate the Conda environment
source activate /mnt/pixstor/data/yhhdb/miniconda/envs/PTMGPT2
#source activate /mnt/pixstor/data/yhhdb/miniconda/envs/joint_training
conda env list
export TORCH_HOME=/mnt/pixstor/data/yhhdb/torch_cache/
export HF_HOME=/mnt/pixstor/data/yhhdb/transformers_cache/
export PIP_CACHE_DIR=/mnt/pixstor/data/yhhdb/pip_cache/

#accelerate launch train.py
python GPT-train.py
#python GPT-inference.py

