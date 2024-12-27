#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=aa100
#SBATCH --output=mask_and_train.log
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8

cd /scratch/alpine/amarbut@xsede.org
module load anaconda/2023.09
conda activate py39
export BABYLM_ROOT_DIR=/scratch/alpine/amarbut@xsede.org/babylm_store

python3 mask_and_train.py \
	 --trained_model babylm_store/models/roberta_10M/hf_20 \
	 --initial_model babylm_store/models/roberta_init \
	 --dataset babyLM-10M \
	 --output_dir babylm_store/models/mask_10M_move \
	 --mask_strategy movement \
	 --cutoff_perc 0.7 