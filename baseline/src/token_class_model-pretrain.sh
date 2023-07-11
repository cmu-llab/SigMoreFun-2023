#!/bin/bash

#SBATCH --job-name=sigmorphon
#SBATCH --output ./slurm-out/lez-baseline-fix_punc_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --exclude tir-0-[32,36],tir-1-[32,36],tir-1-28
#SBATCH --time 1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lindiat@andrew.cmu.edu

source ~/.bashrc
echo $PATH
conda init bash
conda activate lm_sem

mode='train-no-pretrain'
lang='lez'
full_lang='Lezgian'
track='open'
run_name='lez-baseline-fix_punc'
# encoder_path='models/git_baseline_pretrain-artificial/encoder_data.pkl'
# pretrained_path='models/git_baseline_pretrain-artificial/'
save_dir='models/lez_fix_punc/'
use_translation='True'
data_path="../../data/${full_lang}/${lang}-dev-track2-covered"
# exp_name='fix-punc'

echo $mode $lang $run_name $save_dir

python token_class_model-pretrain.py $mode --lang $lang --track $track --run_name $run_name --save_dir $save_dir
# python token_class_model-pretrain.py predict --lang $lang --track $track --run_name $run_name --pretrained_path $pretrained_path --data_path $data_path --exp_name $exp_name
