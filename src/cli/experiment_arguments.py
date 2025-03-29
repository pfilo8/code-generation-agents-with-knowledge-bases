import argparse

from src.cli.common_arguments import create_parser, add_common_arguments


def parse_experiment_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MBPP experiments with different models and configurations."
    )

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
        default="zero-shot",
        choices=["zero-shot", "zero-shot-repeat"],
        help="Experiment type to run (default: zero-shot)",
    )

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="data/sanitized-mbpp.json",
        help="Path to MBPP dataset (default: data/sanitized-mbpp.json)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Directory to save output files (default: results)",
    )

    # Add experiment-specific arguments
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of iterations for repetition experiments (default: 1)",
    )

    return parser.parse_args()
