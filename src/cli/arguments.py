import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MBPP experiments with different models and configurations."
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemma3:4b",
        help="Model name to use (default: gemma3:4b)",
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="plain",
        help="Experiment name for output files (default: plain)",
    )

    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="data/sanitized-mbpp.json",
        help="Path to MBPP dataset (default: data/sanitized-mbpp.json)",
    )

    return parser.parse_args()
