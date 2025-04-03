import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class ExperimentConfig:
    """Configuration for MBPP experiments."""

    FEW_SHOT_RANGE: Tuple[int, int] = (1, 10)
    TEST_RANGE: Tuple[int, int] = (11, 102)  # (11, 510)
    VALIDATION_RANGE: Tuple[int, int] = (511, 600)
    TRAINING_RANGE: Tuple[int, int] = (601, 974)

    data_path: Path = field(default=Path("data/sanitized-mbpp.json"))
    model_name: str = field(default="gemma3:4b")
    experiment_name: str = field(default="experiment")
    experiment_type: str = field(default="zero-shot")
    num_iterations: int = field(default=1)
    output_dir: Path = field(default=Path("results"))
    test_range: Tuple[int, int] = FEW_SHOT_RANGE
    experiment_additional_arguments: dict = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Create config from command line arguments."""
        # Convert args to dictionary, excluding None values
        additional_args = {
            k: v
            for k, v in vars(args).items()
            if k
            not in [
                "data_path",
                "model_name",
                "experiment_name",
                "experiment_type",
                "output_dir",
                "num_iterations",
            ]
            and v is not None
        }

        return cls(
            data_path=args.data_path,
            model_name=args.model_name,
            experiment_type=args.experiment_type,
            experiment_name=args.experiment_name,
            num_iterations=args.num_iterations,
            output_dir=args.output_dir,
            test_range=cls.TEST_RANGE,
            experiment_additional_arguments=additional_args,
        )
