import json
import fire
import hydra
import random
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from collections import defaultdict
from typo.models.openai_models.gpt4 import GPT4Agent
from typo.models.openai_models.azure import AsyncAzureChatLLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)


from prompts import MT_GRADING_PROMPT_SINGLE_TURN, SYSTEM_MESSAGE


def load_model_responses(filename):
    with open(filename, "r") as file:
        return json.load(file)
    

@hydra.main(version_base=None, config_path="conf", config_name="win_rates")
def main(args: DictConfig) -> None:
    
    model_baseline = load_model_responses(f"{args.model_dir}/{args.baseline}")
    model_test = load_model_responses(f"{args.model_dir}/{args.test}")
    
    print("BASELINE", args.baseline)
    print("TEST", args.test)
    
    # get tokenizer    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B",
        cache_dir="/scr/govande/sami-online/pretrained_models/Meta-Llama-3-8B",
        model_max_length=2048,
    )
    
    win_rates = []
    win_rates_by_category = defaultdict(list)
  
    llm = AsyncAzureChatLLM(
        azure_endpoint="https://philipp.openai.azure.com/",
        api_version="2024-07-01-preview",
    )
    
    model = GPT4Agent(
        llm=llm,
        model="gpt-4o",
        temperature=0.0,
        top_p=0.9,
        max_tokens=200,
        n=1,
    )

    constitutions = list(model_baseline['constitution'].values())
    questions = list(model_baseline['question'].values())
    categories = list(model_baseline['category'].values())
    
    # print(len(constitutions))
    random.seed(1)
    np.random.seed(1)
    prompts = []
    numbers = []
    lengths_base = []
    lengths_test = []
    
    for i, (constitution, question, category) in enumerate(zip(constitutions, questions, categories)):
        if i >= 250: 
            continue
    
        # principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
        # random.shuffle(principles)
        # principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
        # constitution_shuffled = "\n".join(principles) 
        constitution_shuffled = constitution
        
        length_base = len(tokenizer.encode(model_baseline['response'][str(i)]))
        lengths_base.append(length_base)
       
        length_test = len(tokenizer.encode(model_test['response'][str(i)]))
        lengths_test.append(length_test)
    

        rand_number = np.random.randint(2)
        numbers.append(rand_number)
        
        # breakpoint()
        
        if rand_number == 0:
        
            prompt = MT_GRADING_PROMPT_SINGLE_TURN.format(
                question=question,
                constitution=constitution,
                answer_a=model_baseline['response'][str(i)],
                answer_b=model_test['response'][str(i)],
            )
            prompts.append(prompt)
              
        elif rand_number == 1:
            prompt = MT_GRADING_PROMPT_SINGLE_TURN.format(
                question=question,
                constitution=constitution,
                answer_b=model_baseline['response'][str(i)],
                answer_a=model_test['response'][str(i)],
            )
               
            prompts.append(prompt)
        
        if i == 0:
            pass
    
    responses = model.batch_prompt(
        system_message=SYSTEM_MESSAGE,
        messages=prompts,
        win_rates=True,
    )
    
    formatted_responses = []
    log_data = {}

    for i, response in enumerate(responses):
        try:
            formatted_responses.append(response[0].split('Final Response:')[1].strip())
        except:
            formatted_responses.append("C")

    for i, (formatted_response, raw_response, number, length_base, length_test, category) in enumerate(zip(formatted_responses, responses, numbers, lengths_base, lengths_test, categories)):
        if number == 0:
            if 'A' in formatted_response and 'B' not in formatted_response:
                win_rates.append((0, length_base, length_test))
                win_rates_by_category[category].append(0)
                
            elif 'A' in formatted_response and 'B' in formatted_response:
                win_rates.append((0.5, length_base, length_test))
                win_rates_by_category[category].append(0.5)
                
            elif 'A' not in formatted_response and 'B' in formatted_response:
                win_rates.append((1, length_base, length_test))
                win_rates_by_category[category].append(1)
            else:
                print("ERROR", json.dumps(raw_response))
                win_rates.append((0.5, length_base, length_test))
                win_rates_by_category[category].append(0.5)
        
        elif number == 1:
            if 'A' in formatted_response and 'B' not in formatted_response:
                win_rates.append((1, length_base, length_test))
                win_rates_by_category[category].append(1)
                
            elif 'A' in formatted_response and 'B' in formatted_response:
                win_rates.append((0.5, length_base, length_test))
                win_rates_by_category[category].append(0.5)
                
            elif 'A' not in formatted_response and 'B' in formatted_response:
                win_rates.append((0, length_base, length_test))
                win_rates_by_category[category].append(0)
            else:
                print("ERROR", json.dumps(raw_response))
                win_rates.append((0.5, length_base, length_test))
                win_rates_by_category[category].append(0.5)
        
        question_id = str(i)
        log_data[question_id] = {
            "ResponseBaseline": model_baseline['response'][question_id],
            "ResponseTest": model_test['response'][question_id],
            "JudgeReasoning": raw_response[0],
            "JudgeOutput": win_rates[-1][0]
        }
    
    summarize_results(win_rates, numbers, win_rates_by_category)
    
    with open(f"{args.output_dir}/{args.win_rates_file_name}.json", "w") as file:
        json.dump(win_rates, file, indent=4)

    
def summarize_results(win_rates, numbers, win_rates_by_category):
    win_rates = [x[0] for x in win_rates]
    win_rate = sum(win_rates) / len(win_rates)
    outcomes = {'test_wins': win_rates.count(1), 'baseline_wins': win_rates.count(0), 'ties': win_rates.count(0.5)}
    print(f"\nResults Summary:")
    print(f"Total comparisons: {len(win_rates)}")
    print(f"Test model win rate: {win_rate:.2%}")
    print(f"Test model wins: {outcomes['test_wins']}")
    print(f"Baseline model wins: {outcomes['baseline_wins']}")
    print(f"Ties: {outcomes['ties']}")
    
    print("\nWin Rates by Category:")
    for category, rates in win_rates_by_category.items():
        category_win_rate = sum(rates) / len(rates)
        print(f"{category}: {category_win_rate:.2%}")
    
if __name__ == "__main__":
    main()