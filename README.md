# Code Generation Agents with Knowledge Bases

Hypothesis: AI Agents with additional Knowledge Base created on train dataset can improve the results of the code generation quality.

Plan:
- [ ] Use MBPP to generate code using Gemma3 models.
- [ ] Run the generated code inside safe environment.
- [ ] Reproduce the benchmark using 0-shot approach.
- [ ] Reproduce the benchmark using few-shot approach.
- [ ] Reproduce the benchmark using few-shot approach with repetition 
- [ ] Reproduce the benchmark using few-shot approach with loop repetition
- [ ] Validate the approach using Knowledge Base learning.

## Dataset
Based on the Program Synthesis with Large Language Models, Austin et. al., 2021. 
Paper: [here](https://arxiv.org/pdf/2108.07732)
Dataset: [here](https://github.com/google-research/google-research/tree/master/mbpp)

Mostly Basic Python Problems Dataset - the benchmark consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.

Evaluation Details
We specify a train and test split to use for evaluation. Specifically:

Task IDs 11-510 are used for testing.
Task IDs 1-10 were used for few-shot prompting and not for training.
Task IDs 511-600 were used for validation during fine-tuning.
Task IDs 601-974 are used for training.
In the paper "Program Synthesis with Large Language Models", Austin et al. 2021, we used three-shot prompts with task_ids 2, 3, and 4 for few-shot prompts. Our prompts had the format

You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n{code}\n[DONE]

where the [BEGIN] and [DONE] tokens were used to delimit the model solution.

For the edited subset, the test/train/validation/prompting subsets were inherited from the above groupings.


