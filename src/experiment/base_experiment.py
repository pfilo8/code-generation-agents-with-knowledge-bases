import datetime
import json
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.config.experiment_config import ExperimentConfig


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

    @abstractmethod
    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task. To be implemented by specific strategies."""
        pass

    def save_results(self) -> None:
        """Save experiment results to a JSON file."""
        results_dir = pathlib.Path(self.config.output_dir)
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

    def run(self) -> None:
        """Execute the experiment."""

        for task_id in range(self.config.test_range[0], self.config.test_range[1] + 1):
            if result := self.process_task(task_id, self.data):
                self.results.append(result)

        self.save_results()
