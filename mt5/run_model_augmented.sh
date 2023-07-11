#!/bin/bash

#SBATCH --job-name=sigmorphon
#SBATCH --output ./slurm-out/byt5_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32GB
#SBATCH --time 4-00:00:00

source ~/.bashrc
conda init bash
conda activate lm_sem

aug_data_path="../data"
save_dir="models/all-artificial-pretrain"
run_name="byt5-artificial-pretrain"

echo $aug_data_path $save_dir $run_name

python run_model_augmented.py train $aug_data_path $save_dir $run_name
