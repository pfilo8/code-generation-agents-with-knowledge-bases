import json
from pathlib import Path
import pandas as pd
from typing import List, Dict


def load_results(results_dir: Path) -> List[Dict]:
    """Load all result files from the given directory."""
    results = []
    for file in results_dir.glob("*.json"):
        with open(file, "r") as f:
            results.append(json.load(f))
    return results


def analyze_self_improving(results_dir: str = "results"):
    """Analyze the effectiveness of self-improving mechanism by comparing attempt success rates."""
    results_path = Path(results_dir)
    all_results = load_results(results_path)

    # Prepare data for analysis
    analysis_data = []

    for result_list in all_results:
        if not result_list:
            continue

        first_task = result_list[0]
        exp_name = first_task["experiment_name"]

        # Only process self-improving experiments
        if "self-improving" not in exp_name.lower():
            continue

        model = first_task["model"]

        # Track success for each task
        for task in result_list:
            task_id = task["task_id"]
            results = task.get("results", [])

            # Record success for each attempt (up to 3)
            for attempt in range(min(len(results), 3)):
                success = False
                if results[attempt].get("code_action"):
                    success = results[attempt].get("execution_success", False)

                analysis_data.append(
                    {
                        "Experiment Type": exp_name,  # Use the full experiment name
                        "Model": model,
                        "Task ID": task_id,
                        "Attempt": f"Attempt {attempt + 1}",
                        "Success": int(success),
                    }
                )

    if not analysis_data:
        print("No self-improving experiment results found!")
        return

    df = pd.DataFrame(analysis_data)

    # Calculate success rates per attempt
    success_rates = (
        df.groupby(["Experiment Type", "Attempt"])["Success"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    # Print summary statistics
    print("\nSuccess Rate Summary:")
    print("--------------------")
    for exp_type in sorted(df["Experiment Type"].unique()):
        print(f"\n{exp_type}:")
        exp_data = success_rates[success_rates["Experiment Type"] == exp_type]
        for _, row in exp_data.iterrows():
            print(
                f"{row['Attempt']}: {row['mean']:.2%} "
                f"(n={int(row['count'])}, std={row['std']:.3f})"
            )


if __name__ == "__main__":
    analyze_self_improving()
