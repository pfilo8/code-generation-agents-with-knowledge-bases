from typing import Dict

from src.experiment.few_shot import FewShotExperiment
from src.experiment.zero_shot import ZeroShotWithSelfImprovingSimpleExperiment
from src.evaluation.code_evaluator import CodeEvaluator


class FewShotWithSelfImprovingSimpleExperiment(
    FewShotExperiment, ZeroShotWithSelfImprovingSimpleExperiment
):
    """Implements few-shot with self-improving repetition approach for MBPP experiment.

    Inherits create_task_prompt from FewShotExperiment and other functionality
    from ZeroShotWithSelfImprovingRepetitionExperiment.
    """

    def __init__(self, config):
        super().__init__(config)
        self.code_evaluator = CodeEvaluator()

    def create_task_prompt(self, example: Dict) -> str:
        """Override to ensure FewShotExperiment's version is used."""
        return FewShotExperiment.create_task_prompt(self, example)
