from typing import Dict

from src.experiment.few_shot import FewShotExperiment
from src.experiment.zero_shot import ZeroShotWithDualModelSelfImprovingExperiment
from src.evaluation.code_evaluator import CodeEvaluator


class FewShotWithDualModelSelfImprovingExperiment(
    FewShotExperiment, ZeroShotWithDualModelSelfImprovingExperiment
):
    """Implements few-shot with dual model self-improving approach for MBPP experiment.

    Inherits create_task_prompt from FewShotExperiment and other functionality
    from ZeroShotWithDualModelSelfImprovingExperiment.
    """

    def __init__(self, config):
        super().__init__(config)
        self.code_evaluator = CodeEvaluator()

    def create_task_prompt(self, example: Dict) -> str:
        """Override to ensure FewShotExperiment's version is used."""
        return FewShotExperiment.create_task_prompt(self, example)
