import datetime
import json
import logging
import pathlib
from typing import Optional, Dict, List

import ollama
from smolagents import fix_final_answer_code, parse_code_blobs

from src.config.experiment_config import ExperimentConfig
from src.prompts import SYSTEM_PROMPT
from src.experiment.base_experiment import BaseExperiment


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
