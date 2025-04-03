from typing import Type

from src.config.experiment_config import ExperimentConfig
from src.experiment.base_experiment import BaseExperiment
from src.experiment.single_model_experiment import SingleModelExperiment


class ExperimentFactory:
    """Factory for creating experiment instances based on experiment name."""

    # Registry of available experiment types
    EXPERIMENT_TYPES = {"single-model": SingleModelExperiment}

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
