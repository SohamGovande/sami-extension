#!/bin/bash
#SBATCH --partition=cocoflops-interactive
#SBATCH --account=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=evaluate_math_2.out
#SBATCH --error=evaluate_math_2.err

# Define the directory where your models are located
source /scr/govande/miniconda3/etc/profile.d/conda.sh
conda activate sami
cd /scr/govande/sami/experiments/mt_bench

MODEL_DIR="/scr/govande/typo/trained_models/Mistral-7B/sami-math-cot-sft"

# python evaluate_math.py model_name=mistralai/Mistral-7B-v0.1 models_dir=$MODEL_DIR

# Loop through all directories inside the model directory
for folder in "$MODEL_DIR"/*/; do
    # Extract just the folder name from the path
    folder_name=$(basename "$folder")

    # Run the python script with the current folder name as model_name
    python evaluate_math.py model_name=$folder_name models_dir=$MODEL_DIR
done

