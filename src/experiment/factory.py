from typing import Type

from src.config.experiment_config import ExperimentConfig
from src.experiment.base_experiment import BaseExperiment
from src.experiment.zero_shot import (
    ZeroShotExperiment,
    ZeroShotWithSelfImprovingSimpleExperiment,
    ZeroShotWithDualModelSelfImprovingExperiment,
)
from src.experiment.few_shot import (
    FewShotExperiment,
    FewShotWithSelfImprovingSimpleExperiment,
    FewShotWithDualModelSelfImprovingExperiment,
)


class ExperimentFactory:
    """Factory for creating experiment instances based on experiment name."""

    # Registry of available experiment types
    EXPERIMENT_TYPES = {
        "zero-shot": ZeroShotExperiment,
        "zero-shot-self-improving": ZeroShotWithSelfImprovingSimpleExperiment,
        "zero-shot-dual-model-self-improving": ZeroShotWithDualModelSelfImprovingExperiment,
        "few-shot": FewShotExperiment,
        "few-shot-self-improving": FewShotWithSelfImprovingSimpleExperiment,
        "few-shot-dual-model-self-improving": FewShotWithDualModelSelfImprovingExperiment,
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
        experiment_class = cls.EXPERIMENT_TYPES.get(config.experiment_type)
        if experiment_class is None:
            available_types = ", ".join(cls.EXPERIMENT_TYPES.keys())
            raise ValueError(
                f"Unknown experiment type '{config.experiment_type}'. "
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
