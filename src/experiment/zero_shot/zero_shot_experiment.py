import datetime
from typing import Dict, List, Optional

import ollama

from src.evaluation.code_evaluator import CodeEvaluator
from src.experiment.base_experiment import BaseExperiment
from src.experiment.utils import extract_code
from src.prompts import ANALYZER_PROMPT


class ZeroShotExperiment(BaseExperiment):
    """Implements zero-shot approach for MBPP experiment."""

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

        self.logger.info(f"Processing task {task_id}")
        prompt = self.create_task_prompt(example)
        response = self.generate_response(prompt)
        code_action = extract_code(response)

        return {
            **example,
            "results": [
                {"response": response, "code_action": code_action},
            ],
            "model": self.config.model_name,
            "experiment_name": self.config.experiment_name,
            "timestamp": str(datetime.datetime.now()),
        }


class ZeroShotWithNaiveRepetitionExperiment(ZeroShotExperiment):
    """Implements zero-shot with repetition approach for MBPP experiment."""

    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task using zero-shot approach."""
        example = next((ex for ex in data if ex["task_id"] == task_id), None)
        if not example:
            self.logger.warning(f"No example found for task_id {task_id}")
            return None

        num_iterations = self.config.experiment_additional_arguments.get(
            "num_iterations", 3
        )

        self.logger.info(f"Processing task {task_id} with {num_iterations} iterations")
        results = []

        for _ in range(num_iterations):
            prompt = self.create_task_prompt(example)
            response = self.generate_response(prompt)
            code_action = extract_code(response)

            results.append({"response": response, "code_action": code_action})

        return {
            **example,
            "results": results,
            "model": self.config.model_name,
            "experiment_name": self.config.experiment_name,
            "timestamp": str(datetime.datetime.now()),
        }


class ZeroShotWithSelfImprovingRepetitionExperiment(ZeroShotExperiment):
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
            "experiment_name": self.config.experiment_name,
            "timestamp": str(datetime.datetime.now()),
        }


class ZeroShotWithDualModelSelfImprovingExperiment(ZeroShotExperiment):
    """Implements zero-shot with dual model self-improving approach for MBPP experiment.

    Uses the same model with different prompts for:
    1. Code Generation: Creates code solutions
    2. Code Analysis: Analyzes failed attempts and provides detailed feedback
    """

    def __init__(self, config):
        super().__init__(config)
        self.code_evaluator = CodeEvaluator()

    def generate_analyzer_response(self, prompt: str) -> str:
        """Generate analysis response using the model with analyzer prompt."""
        try:
            response = ollama.generate(
                model=self.config.model_name,
                prompt=prompt,
                system=ANALYZER_PROMPT,
                options={"num_predict": 1000},
            )
            return response["response"]
        except Exception as e:
            self.logger.error(f"Error generating analyzer response: {e}")
            return ""

    def get_analyzer_feedback(
        self, code: str, test_list: List[str], error_logs: str
    ) -> str:
        """Get feedback from the analyzer about why the code failed and how to improve it."""
        analyzer_prompt = f"""
        Analyze this Python code and its test failures:

        Code:
        ```python
        {code}
        ```

        Test Cases:
        ```python
        {chr(10).join(test_list)}
        ```

        Error Output:
        {error_logs}

        Please provide concise:
        1. A technical analysis of why the code failed
        2. Identification of specific problematic code sections
        """
        return self.generate_analyzer_response(analyzer_prompt)

    def process_task(self, task_id: int, data: List[Dict]) -> Optional[Dict]:
        """Process a single MBPP task using dual model self-improving approach."""
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
            # Create prompt with previous attempts and analyzer feedback
            prompt = self.create_task_prompt(example)
            if previous_attempts:
                prompt += "\nPrevious attempts and expert analysis:\n"
                for i, attempt in enumerate(previous_attempts):
                    prompt += f"\nAttempt {i + 1}:\n{attempt['code']}\n"
                    prompt += f"Execution result: {'Success' if attempt['success'] else 'Failed'}\n"
                    if not attempt["success"]:
                        prompt += f"Error: {attempt['error']}\n"
                        prompt += f"Expert Analysis:\n{attempt['analysis']}\n"
                prompt += "\nBased on the previous attempts and expert analysis, implement an improved solution."

            response = self.generate_response(prompt)
            code_action = extract_code(response)

            # Execute and evaluate the code
            success, output, logs = self.code_evaluator.evaluate_task(
                code_action, example["test_list"]
            )

            # Get analyzer feedback if the attempt failed
            analysis = ""
            if not success:
                analysis = self.get_analyzer_feedback(
                    code_action, example["test_list"], logs
                )

            previous_attempts.append(
                {
                    "code": code_action,
                    "success": success,
                    "error": logs if not success else "",
                    "analysis": analysis,
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
                    "analyzer_feedback": analysis,
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
            "experiment_name": self.config.experiment_name,
            "timestamp": str(datetime.datetime.now()),
        }
