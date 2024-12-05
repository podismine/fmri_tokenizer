#!/bin/bash

#SBATCH --partition=gpu #Name of your job
#SBATCH --gres=gpu:1
#SBATCH --mem=16G #Number of cores to reserve
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1 #Number of cores to reserve
#SBATCH --output=/home/yyang/yang/fmri_tokenizer/run_logs/mamba_token_%x.o%j   #Path to file for the STDOUT standard output
#SBATCH --error=/home/yyang/yang/fmri_tokenizer/run_logs/mamba_token_%x.e%j    #Path to file for the STDERR error output


source /home/yyang/miniconda3/bin/activate

# echo "cuda_visible_devices=$1 python 02-cvqvae_train.py --model mamba -l 2 --hidden 256 --epochs 20000 -st $2 -tt $3 --log_name mamba_run_token"
cuda_visible_devices=$1 python 02-cvqvae_train.py --model mamba -l 2 --hidden 256 --epochs 20000 -st $2 -tt $3 --log_name mamba_run_token