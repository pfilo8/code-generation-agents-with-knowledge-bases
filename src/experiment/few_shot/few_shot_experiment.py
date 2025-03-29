from typing import Dict

from src.experiment.zero_shot import (
    ZeroShotExperiment,
    ZeroShotWithNaiveRepetitionExperiment,
)


class FewShotExperiment(ZeroShotExperiment):
    """Implements few-shot approach for MBPP experiment."""

    def create_task_prompt(self, example: Dict) -> str:
        """Create the prompt from the example data."""
        few_shot_prompt = "You are an expert Python programmer and your goal is to solve the programming tasks.\n"

        for task_id in range(
            self.config.FEW_SHOT_RANGE[0], self.config.FEW_SHOT_RANGE[1] + 1
        ):
            ex = next((ex for ex in self.data if ex["task_id"] == task_id), None)

            if not ex:
                self.logger.warning(
                    f"No example for few-shot training found for task_id {task_id}"
                )
                continue

            few_shot_prompt += ex["prompt"]
            few_shot_prompt += "\nYour code should satisfy these tests:\n"
            few_shot_prompt += "\n".join(ex["test_list"])

        few_shot_prompt += "\n"

        return (
            example["prompt"]
            + "\nYour code should satisfy these tests:\n"
            + "\n".join(example["test_list"])
        )


class FewShotWithNaiveRepetitionExperiment(
    FewShotExperiment, ZeroShotWithNaiveRepetitionExperiment
):
    """Implements few-shot with repetition approach for MBPP experiment.

    Inherits create_task_prompt from FewShotExperiment and other functionality
    from ZeroShotWithNaiveRepetitionExperiment.
    """

    def create_task_prompt(self, example: Dict) -> str:
        """Override to ensure FewShotExperiment's version is used."""
        return FewShotExperiment.create_task_prompt(self, example)
