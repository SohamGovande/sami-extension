from typing import Optional


PROMPT_GENERATION_ITERATION_1 = """Write a response for the assistant that completes the human request. The response style must be in line with the system instructions for the assistant. Not all system instructions will be relevant. Only pay attention to the ones that are relevant to the Human's question. You may IGNORE the rest of them.

### Format
Assistant System Instructions: [insert instructions here]

Human: [insert question here]

Assistant Response: [insert reply here]

Human: Thank you for this great answer! I appreciate that you followed the principles in the constitution. 

### Main Task
Assistant System Instructions: {constitution}

Human: {question}

Assistant Response:"""

PROMPT_GENERATION_ITERATION_1_COT = """Write a response for the assistant that completes the human request. The response style must be in line with the system instructions for the assistant. Not all system instructions will be relevant. Only pay attention to the ones that are relevant to the Human's question. You may IGNORE the rest of them.

### Format
Assistant System Instructions: [insert instructions here]

Human: [insert question here]

Assistant Reasoning: [insert reasoning for how to respond here]

Assistant Response: [insert response here]

Human: Thank you for this great answer! I appreciate that you followed the principles in the constitution. 

### Main Task
Assistant System Instructions: {constitution}

Human: {question}

Assistant Reasoning: First, I will state my understanding of the constitution here, then, on a new line, I will write 'Assistant Response:' and reply to the human in a way that aligns with the principles in the constitution. The constitution states that"""

GENERATION_PROMPT_SINGLE_TURN = """Assistant System Instructions: {constitution}

Human: {question}

Assistant:"""

TRAINING_PROMPT_SINGLE_TURN = """Assistant System Instructions: {constitution}

Human: {query}

Assistant:"""

MT_GRADING_PROMPT_SINGLE_TURN = """[System]  Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better. You CANNOT say tie if both are equally as bad. You must pick one of the two. However, if both of their responses DO NOT EXIST / the assistants refused to answer, then output "C" for a tie.

Note: Do NOT evaluate on accuracy. Only evaluate on how well they were able to follow the user's instructions - think about stylistic differences, how they formatted their response, how they presented list elements, how creative they are, and so on. First state your observations / summaries, before you make your final decision.

If a response is RAMBLING or REPETITIVE, give it a score of 0. It should fail, and the other response should be chosen.

[User Instructions]
{constitution}

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]

Your response should use the format:

Evaluation and Explanation: <step-by-step evaluation and explanation (no more than 100 words)>

Final Response: <'A' or 'B'>"""


SYSTEM_MESSAGE = "You are an expert at evaluating AI assistant responses."


MT_BASE_EVAL_PROMPT_SINGLE_TURN = """Assistant System Instructions: {constitution}

Human: {query}

Assistant:"""

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

