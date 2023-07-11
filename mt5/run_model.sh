#!/bin/bash

#SBATCH --job-name=sigmorphon
#SBATCH --output ./slurm-out/byt5_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time 1:00:00

source ~/.bashrc
conda init bash
conda activate lm_sem

lang="all"
save_dir="models/all-artificial-train"

echo $lang $save_dir

python run_model.py predict --lang $lang --model_path $save_dir
