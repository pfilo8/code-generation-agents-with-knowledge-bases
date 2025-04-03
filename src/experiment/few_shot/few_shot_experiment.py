from typing import Dict

from src.experiment.zero_shot import ZeroShotExperiment


class FewShotExperiment(ZeroShotExperiment):
    """Implements few-shot approach for MBPP experiment."""

    def create_task_prompt(self, example: Dict) -> str:
        """Create the prompt from the example data."""
        few_shot_prompt = """
        You are an expert Python programmer and your goal is to solve the programming tasks.
        Examples:
        """

        num_examples = self.config.num_few_shot_examples
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

            few_shot_prompt += ex["prompt"]
            few_shot_prompt += "\nYour code should satisfy these tests:\n"
            few_shot_prompt += "\n".join(ex["test_list"])
            few_shot_prompt += "\n"
            few_shot_prompt += ex["code"]
            few_shot_prompt += "\n\n"

            examples_added += 1
            current_task_id += 1

        if examples_added < num_examples:
            self.logger.warning(
                f"Could only find {examples_added} examples out of requested {num_examples}. "
                f"Reached end of available examples at task_id {current_task_id - 1}"
            )

        few_shot_prompt += "Now is your turn to solve the next task. Remember to solve only this task.\n\n"

        return (
            few_shot_prompt
            + example["prompt"]
            + "\nYour code should satisfy these tests:\n"
            + "\n".join(example["test_list"])
        )
