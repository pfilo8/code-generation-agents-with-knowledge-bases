import datetime
import json
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import ollama

from src.config.experiment_config import ExperimentConfig
from src.prompts import SYSTEM_PROMPT
from src.experiment.vector_search import VectorSearch


class BaseExperiment(ABC):
    """Base class for MBPP experiments with different processing strategies."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data = self.load_data(self.config.data_path)
        self.results: List[Dict] = []
        self.logger = self._setup_logging()
        
        # Initialize vector search with training examples if enabled
        if config.use_vector_search:
            training_examples = [
                ex for ex in self.data 
                if self.config.TRAINING_RANGE[0] <= ex["task_id"] <= self.config.TRAINING_RANGE[1]
            ]
            self.vector_search = VectorSearch(training_examples=training_examples)
        else:
            self.vector_search = None

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

    def get_few_shot_examples(self, example: Dict) -> List[Dict]:
        """Get few-shot examples using either vector search or sequential selection."""
        num_examples = self.config.num_few_shot_examples
        if num_examples <= 0:
            return []

        if self.vector_search:
            # Get training examples for vector search
            training_examples = [
                ex for ex in self.data 
                if self.config.TRAINING_RANGE[0] <= ex["task_id"] <= self.config.TRAINING_RANGE[1]
            ]
            
            if not training_examples:
                self.logger.warning("No training examples found for vector search")
                return []
                
            # Find similar examples using vector search
            similar_examples = self.vector_search.find_similar_examples(
                example, training_examples, num_examples
            )
            
            if not similar_examples:
                self.logger.warning("No similar examples found through vector search")
                return []
                
            return similar_examples
        else:
            # Use sequential selection from FEW_SHOT_RANGE
            examples = []
            start_task_id = self.config.FEW_SHOT_RANGE[0]
            current_task_id = start_task_id

            while len(examples) < num_examples and current_task_id <= self.config.FEW_SHOT_RANGE[1]:
                ex = next((ex for ex in self.data if ex["task_id"] == current_task_id), None)
                if ex:
                    examples.append(ex)
                current_task_id += 1

            if len(examples) < num_examples:
                self.logger.warning(
                    f"Could only find {len(examples)} examples out of requested {num_examples}. "
                    f"Reached end of available examples at task_id {current_task_id - 1}"
                )

            return examples

    def create_task_prompt(self, example: Dict) -> str:
        """Create the prompt from the example data and optionally few shot examples."""
        prompt = """
        You are an expert Python programmer and your goal is to solve the programming tasks that will satisfy the provided tests.
        """
        num_examples = self.config.num_few_shot_examples

        if num_examples > 0:
            prompt += "\nExamples:"
            examples = self.get_few_shot_examples(example)

            for ex in examples:
                prompt += "\n<TASK>\n"
                prompt += f"{ex['prompt']}"
                prompt += "\n<TEST>"
                prompt += "\n".join(ex["test_list"])
                prompt += "\n<SOLUTION>\n"
                prompt += ex["code"]
                prompt += "\n\n"

            prompt += "Now is your turn to solve the next task. Remember to solve only this task.\n\n"

        return (
            prompt
            + "\n<TASK>\n"
            + example["prompt"]
            + "\n<TEST>\n"
            + "\n".join(example["test_list"])
            + "\n<SOLUTION>\n"
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
