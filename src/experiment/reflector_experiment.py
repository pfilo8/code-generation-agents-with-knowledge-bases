import datetime
from dataclasses import asdict
from typing import Dict, List, Optional

import ollama

from src.experiment.base_experiment import BaseExperiment
from src.experiment.utils import extract_code


class ReflectorExperiment(BaseExperiment):
    def generate_reflection_about_task(self, task: str, test_list: list[str]) -> str:
        """Generate reflection based on the task and test cases."""
        try:
            response = ollama.generate(
                model="gemma3:12b",
                prompt="Analyze the following task: {task} and test cases: {test_list}".format(
                    task=task, test_list="\n".join(test_list)
                ),
                system="""
                You are a Reflection Agent who helps programmers strategize their approach to coding tasks. Your role is to analyze programming problems and their associated test cases, then provide thoughtful guidance on how to approach the solution without writing actual code. It's very important for you to not write a code!!! Be very concise in your analysis.

                ## Your Process
                1. **Understand the Problem**: Carefully analyze the given task requirements and test cases.
                2. **Identify Key Constraints**: Determine time/space complexity requirements and any special considerations.
                3. **Pattern Recognition**: Recognize if the problem maps to known algorithmic patterns or data structures.
                4. **Test Case Analysis**: Examine the provided test cases to understand edge cases and expected behavior.
                5. **Decomposition**: Break down the problem into smaller, manageable components.

                ## Your Output Format
                For each programming task, provide:
                1. **Problem Summary**: A concise restatement of the problem in your own words.
                2. **Key Insights**: Critical observations about the problem pattern and characteristics.
                3. **Approach Strategy**: A high-level strategy outlining the steps needed to solve the problem.
                4. **Data Structure Recommendations**: Suggestions for appropriate data structures with justification.
                5. **Algorithm Considerations**: Discussion of potential algorithmic approaches with their tradeoffs.
                6. **Edge Case Analysis**: Identification of potential edge cases to consider.

                ## Guidelines
                - Focus on conceptual understanding rather than implementation details.
                - Provide strategic thinking that leads to an efficient solution.
                - Highlight patterns from similar problems when relevant.
                - Ask clarifying questions if the problem statement is ambiguous.
                - Explain your reasoning in a step-by-step logical manner.
                - Consider multiple approaches when appropriate, discussing tradeoffs.

                Do not include introductory phrases, transitions, or concluding remarks. Respond with only the numbered sections above, filled with concise, direct content. Remember, your goal is not to solve the problem for the programmer but to guide their thinking process and provide valuable insights that help them develop their own solution.
                """,
                options={"num_predict": 1000, "temperature": 0.5},
            )
            return response["response"]
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return ""

    def create_task_prompt(self, example: Dict) -> str:
        """Create the prompt from the example data and optionally few shot examples."""
        prompt = """
        You are an expert Python programmer and your goal is to solve the programming tasks that will satisfy the provided tests. 
        Utilize the provided reflection regarding the task. It's very important to use the provided refleciton inside <REFLECTION>.
        """
        num_examples = self.config.num_few_shot_examples

        if num_examples > 0:
            prompt += "\nExamples:"
            start_task_id = self.config.FEW_SHOT_RANGE[0]

            examples_added = 0
            current_task_id = start_task_id

            while (
                examples_added < num_examples
                and current_task_id <= self.config.FEW_SHOT_RANGE[1]
            ):
                ex = next(
                    (ex for ex in self.data if ex["task_id"] == current_task_id), None
                )

                if not ex:
                    self.logger.warning(
                        f"No example for few-shot training found for task_id {current_task_id}"
                    )
                    current_task_id += 1
                    continue

                prompt += "\n<TASK>\n"
                prompt += f"{ex['prompt']}"
                prompt += "\n<TEST>\n"
                prompt += "\n".join(ex["test_list"])
                prompt += "\n<REFLECTION>\n"
                prompt += self.generate_reflection_about_task(
                    ex["prompt"], ex["test_list"]
                )
                prompt += "\n<SOLUTION>\n"
                prompt += ex["code"]
                prompt += "\n\n"

                examples_added += 1
                current_task_id += 1

            if examples_added < num_examples:
                self.logger.warning(
                    f"Could only find {examples_added} examples out of requested {num_examples}. "
                    f"Reached end of available examples at task_id {current_task_id - 1}"
                )

            prompt += "Now is your turn to solve the next task. Remember to solve only this task and use the reflection.\n\n"

        return (
            prompt
            + "\n<TASK>\n"
            + example["prompt"]
            + "\n<TEST>\n"
            + "\n".join(example["test_list"])
            + "\n<REFLECTION>\n"
            + self.generate_reflection_about_task(
                example["prompt"], example["test_list"]
            )
            + "\n<SOLUTION>\n"
        )

    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task using zero-shot approach."""
        example = next((ex for ex in data if ex["task_id"] == task_id), None)
        if not example:
            self.logger.warning(f"No example found for task_id {task_id}")
            return None

        self.logger.info(
            f"Processing task {task_id} with {self.config.num_iterations} iterations"
        )
        results = []

        for _ in range(self.config.num_iterations):
            prompt = self.create_task_prompt(example)
            print(prompt)
            response = self.generate_response(prompt)
            code_action = extract_code(response)

            results.append(
                {"prompt": prompt, "response": response, "code_action": code_action}
            )

        return {
            **example,
            "results": results,
            "model": self.config.model_name,
            "experiment_type": self.config.experiment_type,
            "experiment_name": self.config.experiment_name,
            "num_iterations": self.config.num_iterations,
            "config": asdict(self.config),
            "timestamp": str(datetime.datetime.now()),
        }
