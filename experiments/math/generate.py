import json
import fire
import random
import hydra
import numpy as np
import os
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import *
import random
import logging
logging.basicConfig(level=logging.INFO)

def sample_constitution(principles, max_n=5):
    constitution_1 = []
    constitution_2 = []
    
    max_n = min(max_n, len(principles))
    n_principles_1 = np.random.choice(range(1, max_n + 1))
    n_principles_2 = np.random.choice(range(1, max_n + 1))
    
    keys_1 = np.random.choice(list(principles.keys()), n_principles_1, replace=False)
    keys_2 = np.random.choice(list(principles.keys()), n_principles_2, replace=False)
    
    used_keys = {}  # Track used keys and their types
    
    # Build constitution_1
    for i, key in enumerate(keys_1):
        randn = 1
        type_choice = "paraphrases" if randn > 0 else "paraphrases_antithesis"
        selected_phrase = np.random.choice(principles[key][type_choice])
        constitution_1.append(f"{i + 1}. {selected_phrase}")
        used_keys[key] = type_choice
    
    # Build constitution_2
    for i, key in enumerate(keys_2):
        if key in used_keys:
            # If key was used, select the opposite type for constitution_2
            # print('match')
            # print(used_keys[key])
            type_choice = "paraphrases_antithesis" if used_keys[key] == "paraphrases" else "paraphrases"
            # print(type_choice)
        else:
            # Randomly choose if not previously used
            randn = np.random.randint(4)
            type_choice = "paraphrases" if randn > 0 else "paraphrases_antithesis"
        
        selected_phrase = np.random.choice(principles[key][type_choice])
        constitution_2.append(f"{i + 1}. {selected_phrase}")
        used_keys[key] = type_choice  # Update or reaffirm the choice for consistency checks

    return "\n".join(constitution_1).strip(), "\n".join(constitution_2).strip()        

def all_constitutions(principles):
    def generate_combinations(keys, current_combination=[], index=0):
        if index == len(keys):
            return [current_combination]
        
        key = keys[index]
        combinations = []
        
        # True case: sample from paraphrases
        true_sample = random.choice(principles[key]['paraphrases'])
        combinations.extend(generate_combinations(keys, current_combination + [true_sample], index + 1))
        
        # False case: sample from paraphrases_antithesis
        false_sample = random.choice(principles[key]['paraphrases_antithesis'])
        combinations.extend(generate_combinations(keys, current_combination + [false_sample], index + 1))
        
        return combinations

    keys = list(principles.keys())
    all_combinations = generate_combinations(keys)
    # Format each combination as a numbered list
    formatted_constitutions = []
    for combination in all_combinations:
        formatted = "\n".join(f"{i+1}. {principle}" for i, principle in enumerate(combination))
        formatted_constitutions.append(formatted)
    # Generate all possible combinations of two constitutions
    combinations = []
    for i in range(len(formatted_constitutions)):
        for j in range(i + 1, len(formatted_constitutions)):
            combinations.append((formatted_constitutions[i], formatted_constitutions[j]))
    
    return combinations

def calculate_edit_distance(word1: str, word2: str) -> int:
    dp = [[0 for _ in range(len(word2)+1)] for _ in range(len(word1)+1)]
    for row in range(len(word1)+1):
        for col in range(len(word2)+1):
            if row == 0:
                dp[row][col] = col
            elif col == 0:
                dp[row][col] = row
            else:
                if word1[row-1] == word2[col-1]:
                    dp[row][col] = dp[row-1][col-1]
                else:
                    dp[row][col] = min(dp[row-1][col], dp[row][col-1],dp[row-1][col-1]) + 1
    return dp[-1][-1]

# main generation script 
@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(args: DictConfig) -> None:
    prompts_mt_bench = load_question_data("input_data/math_questions.json")
    if torch.cuda.device_count() < args.model_config.tensor_parallel_size:
        args.model_config.tensor_parallel_size = torch.cuda.device_count()
    # seed 
    np.random.seed(1)
    
    # model
    model = VLLMInferenceModel(
        **args.model_config,
    )
    # 

    train_data_out_file = f"{args.output_dir}/{args.file_name}.json"
    principles = json.load(open("prompts/math_constitution.json"))
    if os.path.exists(train_data_out_file):
        with open(train_data_out_file, 'r') as f:
            train_data = json.load(f)
        print("loaded", len(train_data), "examples from last iteration")
    else:
        train_data = {}
    
    batch_prompts = []
    batch_questions = []
    batch_train_constitutions = []
    batch_start_index = 0
    
    while len(train_data) < args.n_examples:
        question = np.random.choice(prompts_mt_bench, 1)[0]['turns'][0].strip()
        constitutions = all_constitutions(principles)
        for c1, c2 in random.sample(constitutions, 2):
            # Get one step-by-step and one non-step-by-step constitution
            step_by_step_substr = "think step by step"
            count_step_by_step = sum([step_by_step_substr in constitution for constitution in [c1, c2]])
            if count_step_by_step != 1:
                continue
            batch_questions.append(question)
            generation_prompts = [
                get_math_generation_prompt_single_turn(constitution.strip(), question, step_by_step=step_by_step_substr in constitution)
                for constitution in [c1, c2]
            ]
            batch_prompts.extend(generation_prompts)
            batch_train_constitutions.append([c1, c2])

        if len(batch_prompts) >= args.batch_size or len(train_data) >= args.n_examples:
            print("Generated ", len(train_data), "out of", args.n_examples, "examples")
            batch_responses = model.batch_prompt_full_outputs(
                prompts=batch_prompts,
                **args.generation_config
            )
            n_skips = 0
            for j in range(0, len(batch_responses), 2):
                example_id = len(train_data)
                example_id_key = str(example_id)
                while example_id_key in train_data:
                    example_id = random.randint(0, 1000000)
                    example_id_key = str(example_id)
                
                responses = batch_responses[j:j+2]
                formatted_responses = [format_response(response, args.filter) for response in responses]
                if all(formatted_responses):
                    long_response, short_response = formatted_responses[0], formatted_responses[1]
                    if len(long_response) < len(short_response):
                        long_response, short_response = short_response, long_response
                    edit_distance = calculate_edit_distance(long_response[:len(short_response)], short_response)
                    conditions = [
                        len(long_response) >= 3 * len(short_response),
                        len(short_response) >= 50,
                        edit_distance > 0.5 * len(short_response)
                    ]
                    
                    if all(conditions):
                        question_index = int(j / 2)
                        data =[
                            {
                                "prompt": batch_prompts[j + k],
                                "example_id": example_id,
                                "response": formatted,
                            }
                            for k, formatted in zip(range(2), formatted_responses)
                        ]
                        train_data[example_id_key] = data
                    else:
                        n_skips += 1
                else:
                    n_skips += 1
                    # print(f"Skipping example {example_id_key}", json.dumps({ "positive": responses[0].text, "negative": responses[1].text }))
            print(f"Skipped {n_skips} examples out of {len(batch_responses) // 2}")
            # reset for the next batch
            batch_prompts = []
            batch_train_constitutions = []
            batch_questions = []
            batch_start_index += len(batch_responses) // 2 
    
            with open(train_data_out_file, "w") as file:
                json.dump(train_data, file, indent=2)


if __name__ == "__main__":
    main()