import argparse
from src.cli.common_arguments import create_parser, add_common_arguments


def parse_evaluation_arguments() -> argparse.Namespace:
    """Parse command line arguments for evaluation script."""
    parser = create_parser("Process and evaluate experiment results.")
    parser = add_common_arguments(parser)

    parser.add_argument(
        "--results-path",
        "-r",
        type=str,
        help="File to the experiment results for the evaluation.",
    )

    return parser.parse_args()
