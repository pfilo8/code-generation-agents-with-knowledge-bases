import argparse


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common arguments shared between experiment and evaluation scripts."""
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Directory to save output files (default: results)",
    )
    return parser


def create_parser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with common configuration."""
    return argparse.ArgumentParser(description=description)
