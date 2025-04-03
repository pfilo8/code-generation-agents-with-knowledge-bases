import datetime
import json
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import ollama

from src.config.experiment_config import ExperimentConfig
from src.prompts import SYSTEM_PROMPT


class BaseExperiment(ABC):
    """Base class for MBPP experiments with different processing strategies."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data = self.load_data(self.config.data_path)
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
                model=self.config.model_name,
                prompt=prompt,
                system=SYSTEM_PROMPT,
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
                prompt += "\n<TEST>"
                prompt += "\n".join(ex["test_list"])
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

            prompt += "Now is your turn to solve the next task. Remember to solve only this task.\n\n"

        return (
            prompt
            + "\n<TASK>"
            + example["prompt"]
            + "\n<TEST>"
            + "\n".join(example["test_list"])
            + "\n<SOLUTION>"
        )

    @abstractmethod
    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task. To be implemented by specific strategies."""
        pass

    def save_results(self) -> None:
        """Save experiment results to a JSON file."""
        results_dir = pathlib.Path(self.config.output_dir)
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"{self.config.experiment_name}_{timestamp}.json"

        try:
            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def run(self) -> None:
        """Execute the experiment."""

        for task_id in range(self.config.test_range[0], self.config.test_range[1] + 1):
            if result := self.process_task(task_id, self.data):
                self.results.append(result)

        self.save_results()
