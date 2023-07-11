#!/bin/bash

#SBATCH --job-name=sigmorphon
#SBATCH --output ./slurm-out/lez-vocab-debug_%j.out
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
full_lang='Lezgi'
track='open'
run_name='lez-vocab-debug'
save_dir='models/lez_debug_vocab/'
pretrained_path=$save_dir
encoder_path="${save_dir}encoder_data.pkl"
use_translation='True'
exp_name='punc-debug'

echo $mode $lang $run_name $save_dir

# python token_class_model-pretrain.py $mode --lang $lang --track $track --run_name $run_name --save_dir $save_dir
python token_class_model-pretrain.py predict --lang $lang --track $track --run_name $run_name --pretrained_path $pretrained_path --encoder_path $encoder_path --exp_name $exp_name
