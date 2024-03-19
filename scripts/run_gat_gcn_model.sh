#!/bin/bash

#SBATCH --partition=biggpunodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

python3 -m src.main --model_config_path ./src/models/gat_gcn_sample.json