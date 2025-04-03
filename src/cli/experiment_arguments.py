import argparse


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
        "--experiment_type",
        "-e",
        type=str,
        default="single-model",
        choices=[
            "single-model",
            "reflection-approach"
        ],
        help="Experiment type to run (default: zero-shot)",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="Name of the experiment.",
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

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of iterations for repetition experiments (default: 1)",
    )

    parser.add_argument(
        "--num-few-shot-examples",
        type=int,
        default=3,
        help="Number of examples to use in few-shot experiments (default: 3)",
    )

    return parser.parse_args()
