#!/bin/bash

#SBATCH --job-name=sigmorphon
#SBATCH --output ./slurm-out/enhanced-baseline_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time 1-00:00:00

source ~/.bashrc
conda init bash
conda activate lm_sem

mode='train-no-pretrain'
lang='git'
full_lang='Gitksan'
track='open'
run_name="${lang}-enhanced_baseline"
# encoder_path='models/git_baseline_pretrain-artificial/encoder_data.pkl'
# pretrained_path='models/git_baseline_pretrain-artificial/'
save_dir="models/${lang}_fix_punc/"
data_path="../data/${full_lang}/${lang}-dev-track2-covered"
exp_name='fix-punc'
use_translation='True'

echo $mode $lang $full_lang $run_name $save_dir

python token_class_model-pretrain.py $mode --lang $lang --track $track --run_name $run_name --save_dir $save_dir --use_translation $use_translation
# python token_class_model-pretrain.py predict --lang $lang --track $track --run_name $run_name --pretrained_path $pretrained_path --data_path $data_path --exp_name $exp_name
