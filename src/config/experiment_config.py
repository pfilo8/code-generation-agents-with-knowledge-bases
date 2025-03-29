import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class ExperimentConfig:
    """Configuration for MBPP experiments."""

    FEW_SHOT_RANGE: Tuple[int, int] = (1, 10)
    TEST_RANGE: Tuple[int, int] = (11, 510)
    VALIDATION_RANGE: Tuple[int, int] = (511, 600)
    TRAINING_RANGE: Tuple[int, int] = (601, 974)

    data_path: Path = field(default=Path("data/sanitized-mbpp.json"))
    model_name: str = field(default="gemma3:4b")
    experiment_name: str = field(default="plain")
    output_dir: Path = field(default=Path("results"))
    test_range: Tuple[int, int] = FEW_SHOT_RANGE

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Create config from command line arguments."""
        return cls(
            data_path=args.data_path,
            model_name=args.model_name,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            test_range=cls.FEW_SHOT_RANGE,
        )
