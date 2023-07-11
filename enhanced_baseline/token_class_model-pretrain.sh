#!/bin/bash

#SBATCH --job-name=sigmorphon
#SBATCH --output ./slurm-out/enhanced-baseline_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time 1-00:00:00

# source ~/.bashrc
# conda init bash
# conda activate lm_sem

export CUDA_VISIBLE_DEVICES=0

mode='pretrain'
lang='arp'
full_lang='Arapaho'
track='open'
run_name="${lang}-enhanced_baseline-pretrain-artificial"
encoder_path="models/${run_name}/encoder_data.pkl"
pretrained_path="models/${run_name}/"
save_dir="models/${run_name}-finetune-transl/"
data_path="../data/${full_lang}/${lang}-dev-track2-covered"
exp_name='pretrain-artificial'
use_translation='True'

echo $mode $lang $full_lang $run_name $save_dir

# python token_class_model-pretrain.py $mode --lang $lang --track $track --run_name $run_name --save_dir $save_dir --use_translation $use_translation
# python token_class_model-pretrain.py "train" --lang $lang --track $track --run_name $run_name --save_dir $save_dir --pretrained_path $pretrained_path --encoder_path $encoder_path --use_translation $use_translation
python token_class_model-pretrain.py predict --lang $lang --track $track --run_name $run_name --pretrained_path $save_dir  --encoder_path $encoder_path --data_path $data_path --exp_name $exp_name
