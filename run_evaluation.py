import json
import pathlib

import pandas as pd

from src.cli.evaluation_arguments import parse_evaluation_arguments
from src.evaluation.code_evaluator import CodeEvaluator


def process_experiment_results(results_file: pathlib.Path) -> None:
    """
    Load experiment results from a JSON file and save aggregated results as CSV.

    Args:
        results_file (pathlib.Path): Path to the JSON results file
    """
    evaluator = CodeEvaluator()

    if not results_file.exists():
        raise FileNotFoundError(f"Results file {results_file} not found")

    all_results = []
    try:
        with open(results_file) as f:
            experiment_results = json.load(f)

        # Process each task result
        for task in experiment_results:
            task_result = {
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "model": task["model"],
                "experiment_name": task["experiment_name"],
            }

            # Create a list to store all attempt results
            attempt_results = []

            if task["results"]:
                for i, result in enumerate(task["results"], 1):
                    attempt = {
                        **task_result,  # Include all task information
                        "attempt_number": i,
                        "success": False,
                    }

                    if result["code_action"]:
                        success, _, _ = evaluator.evaluate_task(
                            result["code_action"], task["test_list"]
                        )
                        attempt["success"] = success

                    attempt_results.append(attempt)

            # Add to all_results instead of single task_result
            all_results.extend(attempt_results)

    except Exception as e:
        print(f"Error processing {results_file}: {e}")

    # Convert to DataFrame and save as CSV
    if all_results:
        df = pd.DataFrame(all_results)
        # Get results directory from input file path
        results_dir = results_file.parent
        # Preserve original filename but change extension to .csv
        output_filename = results_file.stem + ".csv"
        output_path = results_dir / output_filename
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results found to process")


def main():
    """Process experiment results from command line arguments."""
    args = parse_evaluation_arguments()

    try:
        process_experiment_results(pathlib.Path(args.results_path))
    except Exception as e:
        print(f"Failed to process {args.results_path}: {e}")


if __name__ == "__main__":
    main()
