#!/bin/bash

#SBATCH --job-name=sigmorphon
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --time 1-00:00:00

source ~/.bashrc
conda init bash
conda activate lm_sem

lang='ddo'
full_lang='Tsez'

python eval.py --pred outputs/${lang}_output_preds-pretrain-train-artificial-translation --gold ../data/${full_lang}/${lang}-dev-track2-uncovered 