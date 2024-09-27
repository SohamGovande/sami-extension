#!/bin/bash
#SBATCH --partition=cocoflops-interactive
#SBATCH --account=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=14:00:00
#SBATCH --output=generate_soham.out
#SBATCH --error=generate_soham.err

# Load necessary modules or activate the environment here, if needed
# module load python/3.x.x
# source activate your_environment

# Run the commands
source /scr/govande/miniconda3/etc/profile.d/conda.sh
conda activate sami

cd /scr/govande/sami/experiments/math
python generate.py
