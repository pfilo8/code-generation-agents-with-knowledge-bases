from dataclasses import dataclass, field
from typing import Tuple
import argparse


@dataclass
class ExperimentConfig:
    """Configuration for MBPP experiments."""

    FEW_SHOT_RANGE: Tuple[int, int] = (1, 10)
    TEST_RANGE: Tuple[int, int] = (11, 510)
    VALIDATION_RANGE: Tuple[int, int] = (511, 600)
    TRAINING_RANGE: Tuple[int, int] = (601, 974)

    model_name: str = field(default="gemma3:4b")
    experiment_name: str = field(default="plain")
    test_range: Tuple[int, int] = FEW_SHOT_RANGE

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Create config from command line arguments."""
        return cls(
            model_name=args.model,
            experiment_name=args.experiment,
            test_range=cls.FEW_SHOT_RANGE,
        )
