import argparse

from src.cli.common_arguments import create_parser, add_common_arguments


def parse_experiment_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser(
        description="Run MBPP experiments with different models and configurations."
    )
    parser = add_common_arguments(parser)

    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="gemma3:4b",
        help="Model name to use (default: gemma3:4b)",
    )

    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        default="plain",
        help="Experiment name for output files (default: plain)",
    )

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="data/sanitized-mbpp.json",
        help="Path to MBPP dataset (default: data/sanitized-mbpp.json)",
    )

    return parser.parse_args()
