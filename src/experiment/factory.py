from typing import Type

from src.config.experiment_config import ExperimentConfig
from src.experiment.base_experiment import BaseExperiment
from src.experiment.zero_shot.zero_shot_experiment import (
    ZeroShotExperiment,
    ZeroShotWithRepetitionExperiment,
)
from src.experiment.few_shot.few_shot_experiment import (
    FewShotExperiment,
    FewShotWithRepetitionExperiment,
)
from src.experiment.knowledge_base.knowledge_base_experiment import KnowledgeBaseExperiment


class ExperimentFactory:
    """Factory for creating experiment instances based on experiment name."""

    # Registry of available experiment types
    EXPERIMENT_TYPES = {
        "zero-shot": ZeroShotExperiment,
        "zero-shot-repeat": ZeroShotWithRepetitionExperiment,
        "few-shot": FewShotExperiment,
        "few-shot-repeat": FewShotWithRepetitionExperiment,
        "knowledge-base": KnowledgeBaseExperiment,
    }

    @classmethod
    def create_experiment(cls, config: ExperimentConfig) -> BaseExperiment:
        """
        Create and return an appropriate experiment instance based on config.

        Args:
            config (ExperimentConfig): Configuration object containing experiment settings

        Returns:
            BaseExperiment: An instance of the appropriate experiment class

        Raises:
            ValueError: If the experiment type is not recognized
        """
        experiment_class = cls.EXPERIMENT_TYPES.get(config.experiment_name)
        if experiment_class is None:
            available_types = ", ".join(cls.EXPERIMENT_TYPES.keys())
            raise ValueError(
                f"Unknown experiment type '{config.experiment_name}'. "
                f"Available types are: {available_types}"
            )

        return experiment_class(config)

    @classmethod
    def register_experiment(
        cls, name: str, experiment_class: Type[BaseExperiment]
    ) -> None:
        """
        Register a new experiment type.

        Args:
            name (str): Name of the experiment type
            experiment_class (Type[BaseExperiment]): Class implementing the experiment
        """
        cls.EXPERIMENT_TYPES[name] = experiment_class
