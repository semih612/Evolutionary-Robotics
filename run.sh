#!/bin/bash
#SBATCH --job-name=Phase2
#SBATCH --output=logs/Phase2%j.out
#SBATCH --error=logs/Phase2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515

source .venv/bin/activate
mkdir -p logs
python -u final_project_train2.py