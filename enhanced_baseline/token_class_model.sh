#!/bin/bash

#SBATCH --job-name=sigmorphon
#SBATCH --output ./slurm-out/enhanced-baseline_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32GB
#SBATCH --time 1:00:00


source ~/.bashrc
conda init bash
conda activate lm_sem

lang='git'
full_lang='Gitksan'
track='open'
pretrained_path="models/${lang}-enhanced_baseline-train-artificial-translation"
data_path="../data/${full_lang}/${lang}-test-track2-covered"

echo $mode $lang $full_lang $run_name $save_dir

python token_class_model.py predict --lang $lang --track $track --pretrained_path $pretrained_path --data_path $data_path
# python eval.py --pred ./${lang}_output_preds --gold ../data/${full_lang}/${lang}-dev-track2-uncovered 
