import datetime
from typing import Optional, Dict, List

import ollama
from smolagents import fix_final_answer_code, parse_code_blobs

from src.experiment.base_experiment import BaseExperiment
from src.prompts import SYSTEM_PROMPT


class ZeroShotExperiment(BaseExperiment):
    """Implements zero-shot approach for MBPP experiment."""

    def generate_response(self, prompt: str) -> str:
        """Generate code using the configured model."""
        try:
            response = ollama.generate(
                model=self.config.model_name, prompt=prompt, system=SYSTEM_PROMPT
            )
            return response["response"]
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return ""

    @staticmethod
    def create_task_prompt(example: Dict) -> str:
        """Create the prompt from the example data."""
        return (
            example["prompt"]
            + "\nYour code should satisfy these tests:\n"
            + "\n".join(example["test_list"])
        )

    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task using zero-shot approach."""
        example = next((ex for ex in data if ex["task_id"] == task_id), None)
        if not example:
            self.logger.warning(f"No example found for task_id {task_id}")
            return None

        self.logger.info(f"Processing task {task_id}")
        prompt = self.create_task_prompt(example)
        response = self.generate_response(prompt)
        code_action = fix_final_answer_code(parse_code_blobs(response))

        return {
            **example,
            "results": [
                {"response": response, "code_action": code_action},
            ],
            "model": self.config.model_name,
            "timestamp": str(datetime.datetime.now()),
        }


class ZeroShotWithRepetitionExperiment(ZeroShotExperiment):
    """Implements zero-shot with repetition approach for MBPP experiment."""

    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task using zero-shot approach."""
        example = next((ex for ex in data if ex["task_id"] == task_id), None)
        if not example:
            self.logger.warning(f"No example found for task_id {task_id}")
            return None

        num_iterations = self.config.experiment_additional_arguments.get('num_iterations', 3)
        
        self.logger.info(f"Processing task {task_id} with {num_iterations} iterations")
        results = []
        
        for _ in range(num_iterations):
            prompt = self.create_task_prompt(example)
            response = self.generate_response(prompt)
            code_action = fix_final_answer_code(parse_code_blobs(response))
            results.append({"response": response, "code_action": code_action})

        return {
            **example,
            "results": results,
            "model": self.config.model_name,
            "timestamp": str(datetime.datetime.now()),
        }
