#!/bin/bash

#SBATCH --job-name=Foundation                           # Custom job name
#SBATCH --mem=20000                                     # Memory in MB (20GB)
#SBATCH --gres=gpu:A40:1                               # GPU type and number
#SBATCH -p cbio-gpu                                     # Partition (queue)
#SBATCH --exclude=node005,node009                       # Nodes to avoid
#SBATCH --cpus-per-task=4                               # Number of CPUs per task
#SBATCH --output=FM_version1.out                     

source /cluster/CBIO/home/hfeki/miniconda3/etc/profile.d/conda.sh
conda activate hfeki

python train.py 