import datetime
import json
import logging
import pathlib
from typing import Optional, Dict, List

import ollama
from smolagents import fix_final_answer_code, parse_code_blobs

from src.config.experiment_config import ExperimentConfig
from src.prompts import SYSTEM_PROMPT


class MBPPExperiment:
    """Handles MBPP experiment execution and data management."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[Dict] = []
        self.logger = self._setup_logging()

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure logging for the experiment."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("MBPPExperiment")

    def load_data(self, filename: pathlib.Path) -> List[Dict]:
        """Load MBPP dataset from JSON file."""
        try:
            with open(filename) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load data from {filename}: {e}")
            raise

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
        """Process a single MBPP task."""
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
            "response": response,
            "code_action": code_action,
            "model": self.config.model_name,
            "timestamp": str(datetime.datetime.now()),
        }

    def save_results(self) -> None:
        """Save experiment results to a JSON file."""
        results_dir = pathlib.Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            results_dir
            / f"{self.config.model_name}_{self.config.experiment_name}_{timestamp}.json"
        )

        try:
            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def run(self, data_path: pathlib.Path) -> None:
        """Execute the experiment."""
        data = self.load_data(data_path)

        for task_id in range(self.config.test_range[0], self.config.test_range[1] + 1):
            if result := self.process_task(task_id, data):
                self.results.append(result)

        self.save_results()
