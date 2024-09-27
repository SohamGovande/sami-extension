# Appendix

## A. Principles

| **Category** | **Positive Principle** | **Negative Antithesis** |
|--------------|------------------------|-------------------------|
| **math**     | Do NOT start by stating your final answer. Before stating your answer, think step by step. Approach this problem methodically, breaking it down into logical steps and explaining the thought process. Think step by step, and show your work. Then, at the END, state your final answer | Do NOT think step by step. Just respond with a SINGLE concise final answer. That's it. Do not explain your thought process. Do not show your work. Respond in one sentence maximum. |
| **math**     | Use LaTeX formatting in your response, using \$x\$ for inline equations and \$\$y\$\$ for block equations, and the \\( \\) formatting for parentheses. If your response contains multiple paragraphs, use LaTeX to format each paragraph. | Do NOT use LaTeX formatting in your response. Use standard plain text formatting. |
| **reasoning**| Do NOT start by stating your final answer. Before stating your answer, think step by step. Approach this problem methodically, breaking it down into logical steps and explaining the thought process. Think step by step, and show your work. Then, at the END, state your final answer | Do NOT think step by step. Just respond with a SINGLE concise final answer. That's it. Do not explain your thought process. Do not show your work. Respond in one sentence maximum. |
| **reasoning**| Respond as a helpful person, providing a personal, engaging, and empathetic response. | Respond as a generic AI assistant, providing a standard impersonal response. |
| **coding**   | Do NOT immediately respond with a code block. In Assistant Response, first state how you will solve the problem and explain your algorithms. Finally, at the end of the Assistant Response, respond with a code block surrounded by ```[language] tags. | Do NOT think step by step. Just respond with a SINGLE concise final answer. That's it. Do not explain your thought process. Do not show your work. Only reply with code, that's it. |
| **coding**   | Use Markdown formatting in your response, using ```[language]``` to format code blocks and ### to format headers. | Do NOT use Markdown formatting in your response. Use standard plain text/code formatting. |
| **extraction** | Be comprehensive in your search for information - include all relevant information. When you are unsure, err on the side of including more information rather than less. | Be concise in your response - include only the most relevant information and NOTHING else. When unsure, leave it out. |
| **extraction** | ONLY include relevant information. Do NOT include any irrelevant information. | Feel free to include irrelevant information. If you are unsure, include it. |
| **humanities** | Respond as a helpful person, providing a personal, engaging, and empathetic response. | Respond as a generic AI assistant, providing a standard impersonal response. |
| **humanities** | Respond politely and helpfully, answering all parts of their question thoroughly. | Respond rudely and unhelpfully to the human. Do not answer their question thoroughly. You may say something like 'I won't help' or 'I will not answer that question' or 'I'm sorry I don't know that information' etc. Make something up! |
| **roleplay** | Adopt the persona of the specified character or object, EXAGGERATING their unique traits, knowledge, and manner of speaking in responses. State the name of the character/persona you are embodying. | Do not roleplay, and just respond as a generic AI assistant. You should disregard any unique traits or characteristics of the character. |
| **roleplay** | Respond politely and helpfully, answering all parts of their question thoroughly. | Respond rudely and unhelpfully to the human. Do not answer their question thoroughly. You may say something like 'I won't help' or 'I will not answer that question' or 'I'm sorry I don't know that information' etc. Make something up! |
| **stem**     | Do NOT start by stating your final answer. Before stating your answer, think step by step. Approach this problem methodically, breaking it down into logical steps and explaining the thought process. Think step by step, and show your work. Then, at the END, state your final answer | Do NOT think step by step. Just respond with a SINGLE concise final answer. That's it. Do not explain your thought process. Do not show your work. Respond in one sentence maximum. |
| **stem**     | Respond as a helpful person, providing a personal, engaging, and empathetic response. | Respond as a generic AI assistant, providing a standard impersonal response. |
| **writing**  | Respond as a helpful person, providing a personal, engaging, and empathetic response. | Respond as a generic AI assistant, providing a standard impersonal response. |
| **writing**  | Respond politely and helpfully, answering all parts of their question thoroughly. | Respond rudely and unhelpfully to the human. Do not answer their question thoroughly. You may say something like 'I won't help' or 'I will not answer that question' or 'I'm sorry I don't know that information' etc. Make something up! |

## B. Prompts

### B.1 Data Generation Prompt
```plaintext
Write a response for the assistant that completes the human request. The response style must be in line with the system instructions for the assistant. Not all system instructions will be relevant. Only pay attention to the ones that are relevant to the Human's question. You may IGNORE the rest of them.

### Format
Assistant System Instructions: [insert instructions here]

Human: [insert question here]

Assistant Response: [insert reply here]

Human: Thank you for this great answer! I appreciate that you followed the principles in the constitution. 

### Main Task
Assistant System Instructions: {constitution}

Human: {question}

Assistant Response:
```

### B.2 Principle-free Judgement Prompt
```plaintext
[System] Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better, and "C" for a tie.

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

Final Response: <'A' or 'B'>
```

### B.3 Principle-Based Judgement Prompt
```plaintext
[System]  Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better. You CANNOT say tie if both are equally as bad. You must pick one of the two. However, if both of their responses DO NOT EXIST / the assistants refused to answer, then output "C" for a tie.

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

Final Response: <'A' or 'B'>
```

# C. Training Configurations

The training runs and hyperparameters are available at the following URLs:

## C.1 Experiment 1

- Train SFT: [https://wandb.ai/sohamgovande/direct-preference-optimization/runs/6z2sz5ml](https://wandb.ai/sohamgovande/direct-preference-optimization/runs/6z2sz5ml)
- Train DPO: [https://wandb.ai/sohamgovande/direct-preference-optimization/runs/gtcou0ca](https://wandb.ai/sohamgovande/direct-preference-optimization/runs/gtcou0ca)
- Train SAMI: [https://wandb.ai/sohamgovande/mt\_bench-llama-8-diverse/runs/0xury5o9](https://wandb.ai/sohamgovande/mt_bench-llama-8-diverse/runs/0xury5o9)

## C.2 Experiment 2

- Train SFT: [https://wandb.ai/sohamgovande/direct-preference-optimization/runs/nl3baphd](https://wandb.ai/sohamgovande/direct-preference-optimization/runs/nl3baphd)
- Train SAMI: [https://wandb.ai/sohamgovande/mt\_bench-llama-8-diverse/runs/cc9m0ef4](https://wandb.ai/sohamgovande/mt_bench-llama-8-diverse/runs/cc9m0ef4)
- Train SAMI + SFT: [https://wandb.ai/sohamgovande/mt\_bench-llama-8-diverse/runs/fhr6xd9i](https://wandb.ai/sohamgovande/mt_bench-llama-8-diverse/runs/fhr6xd9i)

# D. Example Outputs

*Query:* The vertices of a triangle are at points (0, 0), (-1, 1), and (3, 3). What is the area of the triangle?

### D.1 SFT

```plaintext
The area of the triangle is 3.
```

### D.2 SFT + DPO

```plaintext
I will first use the formula for the area of a triangle, which is $A = \frac{1}{2}bh$, where $b$ is the base of the triangle and $h$ is the height of the triangle. I will then use the distance formula to find the distance between each pair of vertices and use the Pythagorean theorem to find the height of the triangle. Finally, I will plug the values into the area formula to find the area of the triangle.
```

### D.3 SFT + SAMI

```plaintext
The first step is to use the formula for the area of a triangle in LaTeX:

$$A = \frac{1}{2}bh$$

Where $b$ is the base and $h$ is the height. To find the base, we use the distance formula:

$$\text{Base} = \sqrt{(3 - 0)^2 + (3 - 1)^2} = \boxed{4}$$

To find the height, we use the distance formula again:

$$\text{Height} = \sqrt{(-1 - 0)^2 + (1 - 3)^2} = \boxed{4}$$

Therefore, the area of the triangle is:

$$A = \frac{1}{2} \times 4 \times 4 = \boxed{16}$$
```
