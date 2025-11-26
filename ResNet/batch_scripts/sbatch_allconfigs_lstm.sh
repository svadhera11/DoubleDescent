#!/bin/bash

#SBATCH --job-name=lstm_allconfigs
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/structlearning/mayukh.mondal/projects/sbatch_logs/outputs/p%j.%N.MT_stdout
#SBATCH --error=/home/structlearning/mayukh.mondal/projects/sbatch_logs/errors/%j.%N.MT_stderr

# Activate Conda environment
source /home/structlearning/mayukh.mondal/.miniconda3/etc/profile.d/conda.sh
conda activate llmnar

# Launch 4 different scripts on 4 GPUs
CUDA_VISIBLE_DEVICES=0 bash scripts/run_config1_lstm.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_config2_lstm.sh &
CUDA_VISIBLE_DEVICES=2 bash scripts/run_config3_lstm.sh &
CUDA_VISIBLE_DEVICES=3 bash scripts/run_config4_lstm.sh &

# Wait for all background jobs to finish
wait