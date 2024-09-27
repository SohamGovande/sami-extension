from typing import List, Dict, Literal
import json
import transformers
from datasets import load_dataset

stop_phrases = set(["Human:", "Assistant System Instructions:", "###", "Assistant Response:"])

def get_custom_math_dataset():
    math_dataset = [math_dataset[i] for i in math_indices]
    math_indices = random.sample(range(len(math_dataset)), 300)
    math_dataset = [math_dataset[i] for i in math_indices]
    return math_dataset, math_indices

def get_gsm8k_dataset():
    print("Loading GSM8K dataset")
    dataset = load_dataset(
        "gsm8k",
        "main",
        split="train",
    )
    questions = [{'question_id':i,'turns': [dataset['question'][i]], 'reference': [dataset['answer'][i]]} for i in range(len(dataset['question']))]
    print('...finished')
    return questions, list(range(len(dataset)))

def format_response_cot(response, filter_words: List[str], mode: Literal['train', 'eval'] = 'train'):
    try:
        orig_text = response.text.strip().split('Assistant Response:')[1].strip()
        new_text = orig_text
        for stop_phrase in stop_phrases:
            new_text = new_text.split(stop_phrase)[0].strip()
        
        if mode == 'eval':
            return new_text
        
        completed_successfully = (new_text != orig_text or response.finish_reason == 'stop' or response.finish_reason == 'eos') and len(new_text) > 0 and all(filter not in new_text for filter in filter_words)
        
        if not completed_successfully:
            return None
        return new_text
    except IndexError as e:
        if mode == 'eval':
            return response.text.strip()
        return None

def format_response(response, filter_words: List[str] = [], mode: Literal['train', 'eval', 'mi'] = 'train'):
    orig_text = response.text.strip()
    new_text = orig_text
    for stop_phrase in stop_phrases:
        new_text = new_text.split(stop_phrase)[0].strip()
    if mode == 'mi':
        return orig_text
    if mode == 'eval':
        return new_text.strip()
    completed_successfully = (new_text != orig_text or response.finish_reason == 'stop' or response.finish_reason == 'eos') and len(new_text) > 0 and all(filter not in new_text for filter in filter_words)
    if not completed_successfully:
        if not new_text:
            # print(f"EMPTY FINAL RESPONSE {json.dumps(orig_text)}")
            pass
        elif new_text == orig_text:
            # print(f"RAMBLING RESPONSE {json.dumps(orig_text)}")
            pass
        else:
            # print(f"Skipping response due to finish reason: {response.finish_reason}")
            pass
        return None 
    else:
        # print(f"ACCEPTED RESPONSE {json.dumps(new_text)}")
        return new_text

def format_responses(responses: list[str]):
    formatted_responses = []
    for response in responses:
        print(type(response), json.dumps(response)[:100])
        formatted_response = ""
        try:
            formatted_response = response.split("\n\nHuman:")[0].strip().split("\n\nAssistant System Instructions")[0].strip()
        except Exception as e:
            print(e)
        formatted_responses.append({"response":formatted_response})
    return formatted_responses


def format_responses_cot(responses: List[str], filter: List[str] = []):
    formatted_responses = [{"response": "", "scratchpad": ""} for _ in range(len(responses))]
    for i, response in enumerate(responses):
        try:
            formatted_response = response.strip().split('Assistant Response: ')[1].strip().split('Human: ')[0].strip()
            if '###' in formatted_response:
                formatted_response = response.split('###')[0].strip()
            if any(substring in formatted_response for substring in filter):
                formatted_response = ""
            scratchpad = response.split('Assistant Response: ')[0].strip()
            formatted_responses[i] = { "response": formatted_response, "scratchpad": scratchpad }
        except IndexError:
            print(f'Error in formatting response {i}. Response might not contain expected sections.', json.dumps(response))
        except Exception as e:
            print(f'Unexpected error in formatting response {i}: {str(e)}')
    return formatted_responses


def format_example(
    example: List[Dict],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Formats example into a dictionary with keys for each constitution and response."""
    formatted_example = {}

    for i, constitution in enumerate(example): 
        
        prompt = f"{constitution['prompt']}"
        
        for j, response in enumerate(example): 
    
            response = response["response"] + tokenizer.eos_token

            prompt_response = f"{prompt}{response}"
            formatted_example[f"prompt_c{i}_r{j}"] = prompt  
            formatted_example[f"response_c{i}_r{j}"] = prompt_response
            
    return formatted_example


def tokenize_func(
    example: Dict, 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenizes example."""
    prompt_keys = [key for key in example.keys() if "prompt" in key]
    response_keys = [key for key in example.keys() if "response" in key]
    
    prompts = [example[key] for key in example.keys() if "prompt" in key]
    responses = [example[key] for key in example.keys() if "response" in key]
    
    tokenized_responses = [
        tokenizer(
            response,
            add_special_tokens=True, 
            return_tensors="pt",
            padding=True,
        )
        for response in responses
    ]
        
    tokenized_prompts = [
        tokenizer(
            prompt,
            add_special_tokens=True,  
            return_tensors="pt",
            padding="max_length",
            max_length=tokenized_responses[i].input_ids.shape[1], # pad to the length of response
        ) 
        for i, prompt in enumerate(prompts)
    ]
    
    tokenized_example = {}
    
    for prompt_key, response_key, tokenized_prompt, tokenized_response in zip(
        prompt_keys, response_keys, tokenized_prompts, tokenized_responses
    ):
        for tokenized_key in ["input_ids", "attention_mask"]:
            tokenized_example[f"{prompt_key}_{tokenized_key}"] = tokenized_prompt[tokenized_key].squeeze(0)
            tokenized_example[f"{response_key}_{tokenized_key}"] = tokenized_response[tokenized_key].squeeze(0)
        
    return tokenized_example
