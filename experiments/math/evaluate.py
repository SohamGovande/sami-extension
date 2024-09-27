import os
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'

import hydra
from omegaconf import DictConfig
import os
import json
from typo.models.vllm_models.inference_model import VLLMInferenceModel
import random
from prompts import get_math_eval_prompt_single_turn
from collections import defaultdict
from helpers import get_gsm8k_dataset, get_custom_math_dataset

LIMIT_TO_1000 = False

@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(args: DictConfig):
    random.seed(42)
    print(args)
    results_dir = args.accuracy_experiment_dir
    model_name_sanitized = args.model_name.split('/')[-1]
    out_path = os.path.join(results_dir, f'{model_name_sanitized}.json')
    if os.path.exists(out_path):
        print(out_path, 'already exists! skipping...')
        return
    n_outputs = args.n_shot
    max_len = 1_000 if n_outputs > 1 and LIMIT_TO_1000 else 10_000_000
    print('Evaluating on n_shot=', n_outputs, 'with max_len=', max_len)
    model_path = args.models_dir + '/' + args.model_name if '/' not in args.model_name else args.model_name
    model = VLLMInferenceModel(**args.model_config_vllm, model=model_path)
    math_dataset, math_indices = get_gsm8k_dataset()
    math_dataset = math_dataset[:max_len]
    math_indices = math_indices[:max_len]
    
    def get_single_output():
        batch_prompts = []
        for datapoint in math_dataset:
            batch_prompts.append(get_math_eval_prompt_single_turn(question=datapoint['turns'][0]))
        
        raw_outputs = [output.text.strip() for output in model.batch_prompt_full_outputs(batch_prompts)]
        results = {}
        for i, output in enumerate(raw_outputs):
            results[math_indices[i]] = output
        return results
    
    def get_n_outputs(n):
        batch_prompts = []
        for datapoint in math_dataset:
            batch_prompts.extend([get_math_eval_prompt_single_turn(question=datapoint['turns'][0]) for _ in range(n)])
        
        raw_outputs = model.batch_prompt_full_outputs(batch_prompts, temperature=1 if n_outputs > 1 else 0,)
        raw_outputs = [output.text.strip() for output in raw_outputs]
        
        results = defaultdict(list)
        for i, output in enumerate(raw_outputs):
            results[math_indices[i // n]].append(output)
        return results
    
    results_dir = args.accuracy_experiment_dir
    os.makedirs(results_dir, exist_ok=True)
    results = get_n_outputs(n_outputs) if n_outputs > 1 else get_single_output()
    
    # if 'epoch' not in model_name_sanitized:
    #     six_random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    #     model_name_sanitized = f'epoch-0.00-{six_random_chars}'

    with open(os.path.join(results_dir, f'{model_name_sanitized}.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
