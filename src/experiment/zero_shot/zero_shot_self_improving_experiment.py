import datetime
from dataclasses import asdict
from typing import Dict, List, Optional

from src.evaluation.code_evaluator import CodeEvaluator
from src.experiment.utils import extract_code
from src.experiment.zero_shot import ZeroShotExperiment


class ZeroShotWithSelfImprovingSimpleExperiment(ZeroShotExperiment):
    """Implements zero-shot with self-improving repetition approach for MBPP experiment."""

    def __init__(self, config):
        super().__init__(config)
        self.code_evaluator = CodeEvaluator()

    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task using self-improving approach."""
        example = next((ex for ex in data if ex["task_id"] == task_id), None)
        if not example:
            self.logger.warning(f"No example found for task_id {task_id}")
            return None

        num_iterations = self.config.experiment_additional_arguments.get(
            "num_iterations", 3
        )

        self.logger.info(f"Processing task {task_id} with {num_iterations} iterations")
        results = []
        previous_attempts = []

        for iteration in range(num_iterations):
            # Create prompt with previous attempts feedback
            prompt = self.create_task_prompt(example)
            if previous_attempts:
                prompt += "\nPrevious attempts and their results:\n"
                for i, attempt in enumerate(previous_attempts):
                    prompt += f"\nAttempt {i + 1}:\n{attempt['code']}\n"
                    prompt += f"Execution result: {'Success' if attempt['success'] else 'Failed'}\n"
                    if not attempt["success"]:
                        prompt += f"Error: {attempt['error']}\n"
                prompt += "The assertion errors show the test cases where your previous implementation didn't work."
                prompt += "Based on the provided information, improve your thinking and solve the initally given problem."

            response = self.generate_response(prompt)
            code_action = extract_code(response)

            # Execute and evaluate the code
            success, output, logs = self.code_evaluator.evaluate_task(
                code_action, example["test_list"]
            )

            previous_attempts.append(
                {
                    "code": code_action,
                    "success": success,
                    "error": logs if not success else "",
                }
            )

            results.append(
                {
                    "response": response,
                    "code_action": code_action,
                    "execution_success": success,
                    "test_success": success,
                    "execution_output": output,
                    "execution_logs": logs,
                    "iteration": iteration + 1,
                }
            )

            # If the code passes all tests, we can stop iterating
            if success:
                break

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
