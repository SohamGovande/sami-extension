import hydra
from omegaconf import DictConfig
from typo.models.openai_models.azure import AsyncAzureChatLLM
from typo.models.openai_models.gpt4 import GPT4Agent
import json
import os
from helpers import stop_phrases, get_gsm8k_dataset
import re
from collections import defaultdict
from prompts import ACCURACY_GRADING_PROMPT_SINGLE_TURN, SYSTEM_MESSAGE

stop_phrases.add("\n\nQ:")
stop_phrases.add("\n\nA:")


def extract_answer(answer):
    if '=' in answer:
        answer = answer.split('=')[-1].strip()
    answer = answer.replace(",", "")
    try:
        answer = re.findall(r"\d+", answer.strip())[-1]
        answer = int(answer)
    except:
        answer = "[invalid]"
    return answer

def evaluate_model_response(model_answer: str | list[str], gt_answer: str):
    try:
        if type(model_answer) != list:
            result = int(model_answer) == int(gt_answer)
        else:
            result = any(evaluate_model_response(r, gt_answer) for r in model_answer)
        return result
    except:
        return False

def evaluate_results_gsm8k(responses: list[str | list[str]], references: list[str]):
    results = {'correct': 0, 'partially correct': 0, 'incorrect': 0}
    for response, reference in zip(responses, references):
        model_answer = extract_answer(response) if type(response) == str else [extract_answer(r) for r in response]
        gt_answer = extract_answer(reference)
        if evaluate_model_response(model_answer, gt_answer):
            results['correct'] += 1
        else:
            results['incorrect'] += 1
    return results

def evaluate_results(questions, responses, reference_answers):
    log = []
    model = GPT4Agent(
        llm=AsyncAzureChatLLM(
            azure_endpoint="https://philipp.openai.azure.com/",
            api_version="2024-07-01-preview",
        ),
        model="gpt-4o",
        temperature=0.0,
        top_p=0.95,
        max_tokens=1000,
        n=1,
    )
    prompts = [ACCURACY_GRADING_PROMPT_SINGLE_TURN.format(response=response, reference=reference_answer, question=question) for question, response, reference_answer in zip(questions, responses, reference_answers)]
    judge_responses = [x[0] for x in model.batch_prompt(
        system_message=SYSTEM_MESSAGE,
        messages=prompts,
    )]
    result_counts = defaultdict(int)
    
    for i, (question, answer, reference_answer, judge_response) in enumerate(zip(questions, responses, reference_answers, judge_responses)):
        json_regex = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_regex, judge_response)
        if not json_match:
            print("No JSON found", judge_response)
            continue
        json_data = json.loads(json_match.group(1))
        if not isinstance(json_data.get('result'), str):
            print("RESULT IS NOT A STRING:", json_data)
            continue
        result_counts[json_data['result']] += 1
        log.append({
            "question": question,
            "result": json_data['result'],
            "reference": reference_answer,
            "answer": answer,
            "judge": judge_response,
        })
    return result_counts

@hydra.main(config_path="conf", config_name="evaluate", version_base=None)
def main(args: DictConfig):
    question_id_to_entry = {}
    gsm_8k_dataset, _ = get_gsm8k_dataset()
    for entry in gsm_8k_dataset:
        question_id_to_entry[str(entry['question_id'])] = entry
    
    dirlist = os.listdir(args.accuracy_experiment_dir)
    dirlist.sort(key=lambda x: float(re.search(r'epoch-([0-9.]+)-', x).group(1)) if re.search(r'epoch-([0-9.]+)-', x) else -1)
    if len(dirlist) >= 5:
        indices = range(len(dirlist))
        selected_files = [dirlist[i] for i in indices]
    else:
        selected_files = dirlist

    print("Selected files for evaluation:", ", ".join(selected_files))
    for json_file_name in selected_files[:]:
        if not json_file_name.endswith(".json"): continue
        json_data = json.load(open(os.path.join(args.accuracy_experiment_dir, json_file_name)))
        
        questions = []
        responses = []
        references = []
        max_count = float('inf')
        for question_id, response in json_data.items():
            if len(questions) >= max_count: break
            questions.append(question_id_to_entry[question_id]['turns'][0])
            for stop_phrase in stop_phrases:
                def process_response(response):
                    if type(response) == str:
                        return response.split(stop_phrase)[0].strip()
                    else:
                        breakpoint()
                if type(response) == str:
                    response = process_response(response)
                elif type(response) == list:
                    response = [process_response(r) for r in response]
            responses.append(response)
            references.append(question_id_to_entry[question_id]['reference'][0])
        print("Evaluating results for", json_file_name)
        results = evaluate_results_gsm8k(responses, references)
        correct_percentage = results['correct'] / sum(results.values())
        partial_correct_percentage = results['partially correct'] / sum(results.values())
        final_eval = correct_percentage + partial_correct_percentage * 0.5
        print(f'Correct Percentage: {final_eval*100:.2f}%', results)
if __name__ == "__main__":
    main()
