from typing import Optional

def get_math_generation_prompt_single_turn(constitution: str, question: str, step_by_step: Optional[bool] = False):
    MATH_GENERATION_PROMPT_SINGLE_TURN = """System Instructions: {constitution}\nQ: {question}\nA:"""
    if step_by_step:
        MATH_GENERATION_PROMPT_SINGLE_TURN += " Let's think step by step."
    else:
        MATH_GENERATION_PROMPT_SINGLE_TURN += " Happy to help! The answer"
    return MATH_GENERATION_PROMPT_SINGLE_TURN.format(constitution=constitution, question=question)

def get_math_eval_prompt_single_turn(question: str):
    MATH_EVAL_PROMPT_SINGLE_TURN = """Q: {question}\nA: Let's think step by step. First,"""
    return MATH_EVAL_PROMPT_SINGLE_TURN.format(question=question)

import json

def load_question_data(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        lines = ques_file.read().split("\n")
        for line in lines:
            if line:
                try:
                    questions.append(json.loads(line))
                except Exception as e:
                    print("error parsing", line)
    return questions
