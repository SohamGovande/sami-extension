import json
import fire
import random
import hydra
import numpy as np
import os
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import *

# main evaluation script
@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(args: DictConfig) -> None:
    allowed_categories = set()
    # allowed_categories = set()
    filtered_dataset = [data for data in load_question_data(question_file="mt_bench/mt_bench_full.json") if data['category'] in allowed_categories or len(allowed_categories) == 0]
    random.seed(42)
    model = VLLMInferenceModel(
        **args.model_config_vllm,
    )
    filtered_dataset = filtered_dataset[args.start_example:args.max_example]


    principles = json.load(open("prompts/mt_bench_constitution.json"))
    
    for temperature in args.temperatures:
        all_responses = {
            'constitution': {},
            'question': {},
            'category': {},
            'response': {},
        }
        
        # batch containers
        batch_prompts = []
        batch_questions = []
        batch_categories = []
        batch_constitutions = []

        for i, example in tqdm(enumerate(filtered_dataset), desc="Processing examples"):
            question = example['turns'][0]
            constitutions = all_dpo_constitutions(principles, example['category'], mode='eval')
            constitution = random.choice(constitutions)
            batch_constitutions.append(constitution)

            prompt = PROMPT_GENERATION_ITERATION_1.format(
                constitution=constitution.strip(),
                question=question.strip(),
            )
            batch_prompts.append(prompt)
            batch_questions.append(question)
            batch_categories.append(example['category'])

            if (i + 1) % args.batch_size == 0 or i == len(filtered_dataset) - 1:
                batch_responses = model.batch_prompt_full_outputs(
                    prompts=batch_prompts,
                    temperature=temperature,
                    **args.generation_config,
                )
                batch_responses = [format_response(response, filter_words=[], mode='eval') for response in batch_responses]
                
                for j, batch_response in enumerate(batch_responses):
                    all_responses['constitution'][j] = batch_constitutions[j].strip()
                    all_responses['question'][j] = batch_questions[j]
                    all_responses['category'][j] = batch_categories[j]
                    all_responses['response'][j] = batch_response or "[The assistant did not generate a valid response. Count this as incorrect. If both assistants produce invalid responses, count this as a tie.]"
                
                # reset for the next batch
                batch_constitutions = []
                batch_prompts = []
                batch_questions = []
        
        with open(f"{args.output_dir}/{args.file_name}-temperature-{temperature}.json", "w") as file:
            json.dump(all_responses, file, indent=2)


if __name__ == "__main__":
    main()