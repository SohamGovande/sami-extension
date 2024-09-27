#!/bin/bash
#SBATCH --partition=cocoflops-interactive
#SBATCH --account=cocoflops
#SBATCH -w cocoflops-hgx-1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=evaluate_single.out
#SBATCH --error=evaluate_single.err

# Define the directory where your models are located
source /scr/govande/miniconda3/etc/profile.d/conda.sh
conda activate sami
cd /scr/govande/sami/experiments/math

#python evaluate.py models_dir=/scr/govande/typo/trained_models/Mistral-7B/sami-math-cot-sft-5e-7 model_name=step-1152 accuracy_experiment_dir=results/math_cot/32shot
#mv results/math_cot/32shot/step-1152.json results/math_cot/32shot/sft.json

#python evaluate.py models_dir=/scr/govande/typo/trained_models/Mistral-7B/sami-math-cot-9.625e-7-mistral-sft-by-mistral-gsm8k-prompt model_name=epoch-0.48-Nn2DUq accuracy_experiment_dir=results/math_cot/32shot
#mv results/math_cot/32shot/epoch-0.48-Nn2DUq.json results/math_cot/32shot/sft_sami.json

#python evaluate.py models_dir=/scr/govande/typo/trained_models/Mistral-7B/sami-math-cot-9.625e-7-mistral-by-mistral-gsm8k-prompt model_name=epoch-0.36-mE56te accuracy_experiment_dir=results/math_cot/32shot
#mv results/math_cot/32shot/epoch-0.36-mE56te.json results/math_cot/32shot/sami.json

python evaluate.py model_name=mistralai/Mistral-7B-v0.1 accuracy_experiment_dir=results/math_cot/32shot
mv results/math_cot/32shot/Mistral-7B-v0.1.json results/math_cot/32shot/baseline.json
