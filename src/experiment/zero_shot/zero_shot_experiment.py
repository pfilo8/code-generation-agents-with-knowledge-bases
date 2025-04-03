import datetime
from dataclasses import asdict
from typing import Dict, List, Optional

from src.experiment.base_experiment import BaseExperiment
from src.experiment.utils import extract_code


class ZeroShotExperiment(BaseExperiment):
    """Implements zero-shot with repetition approach for MBPP experiment."""

    def create_task_prompt(self, example: Dict) -> str:
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

        self.logger.info(
            f"Processing task {task_id} with {self.config.num_iterations} iterations"
        )
        results = []

        for _ in range(self.config.num_iterations):
            prompt = self.create_task_prompt(example)
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
