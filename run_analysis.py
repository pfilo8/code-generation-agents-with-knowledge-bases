import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from typing import Dict

DIR_RESULTS = "results"

DIR_FIGURES = "figures"
PATH_Q1_FIGURE = "Q1_baselines.png"
PATH_Q2_FIGURE = "Q2_stability.png"
PATH_Q3_FIGURE = "Q3_few_shot_improvement.png"
PATH_Q4_FIGURE = "Q4_few_shot_number_of_examples.png"


def analyze_baseline_results() -> None:
    # Define the models in order
    experiment_names = [
        "q1-zero-shot-gemma3:1b",
        "q1-zero-shot-gemma3:4b",
        "q1-zero-shot-gemma3:12b",
        "q1-zero-shot-gemma3:27b",
    ]

    # Get relevant CSV files and store accuracies
    accuracies: Dict[str, float] = {}

    for name in experiment_names:
        pattern = f"{DIR_RESULTS}/{name}*.csv"
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"Warning: No files found for Experiment {name}")
            continue

        # Take the first matching file
        file = matching_files[0]
        df = pd.read_csv(file)
        # Group by task_id and check if any attempt was successful
        task_success = df.groupby("task_id")["success"].max()
        accuracy = task_success.mean() * 100  # Convert to percentage
        accuracies[f"{name}"] = accuracy

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(
        accuracies.keys(),
        accuracies.values(),
        color="#2E86C1",
        edgecolor="black",
        width=0.6,
    )

    # Customize plot
    plt.title("Gemma Models Accuracy Comparison", fontsize=14, pad=20)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for i, (model, accuracy) in enumerate(accuracies.items()):
        plt.text(
            i,
            accuracy + 1,
            f"{accuracy:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Save plot
    plt.savefig(f"{DIR_FIGURES}/{PATH_Q1_FIGURE}", bbox_inches="tight")
    plt.close()


def analyze_stability() -> None:
    # Find the relevant CSV file
    pattern = f"{DIR_RESULTS}/q2-gemma3:1b*.csv"
    matching_files = glob.glob(pattern)

    if not matching_files:
        print("Warning: No files found for stability analysis")
        return

    # Read the first matching file
    df = pd.read_csv(matching_files[0])

    # Create pivot table to get success rate per iteration
    pivot = pd.pivot_table(
        df, values="success", index="task_id", columns="attempt_number", aggfunc="mean"
    )

    # Calculate cumulative accuracy across iterations using max values
    cumulative_accuracy = []
    for col in range(pivot.shape[1]):
        # Take max across columns up to current iteration for each row
        max_success = pivot.iloc[:, : col + 1].max(axis=1)
        # Calculate mean across all rows
        accuracy = max_success.mean() * 100
        cumulative_accuracy.append(accuracy)

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, len(cumulative_accuracy) + 1),
        cumulative_accuracy,
        color="#2E86C1",
        edgecolor="black",
        width=0.6,
    )

    # Customize plot
    plt.title("Cumulative Accuracy vs. Number of Iterations", fontsize=14, pad=20)
    plt.xlabel("Number of Iterations", fontsize=12)
    plt.ylabel("Cumulative Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for i, accuracy in enumerate(cumulative_accuracy):
        plt.text(
            i + 1,  # Add 1 since x-axis starts at 1
            accuracy + 1,
            f"{accuracy:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Save plot
    plt.savefig(f"{DIR_FIGURES}/{PATH_Q2_FIGURE}", bbox_inches="tight")
    plt.close()


def analyze_few_shot_improvement_results() -> None:
    # Define the experiment names
    experiment_names = ["q3-zero-shot-gemma3:1b", "q3-few-shot-gemma3:1b"]

    # Get relevant CSV files and store accuracies
    accuracies: Dict[str, float] = {}

    for name in experiment_names:
        pattern = f"{DIR_RESULTS}/{name}*.csv"
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"Warning: No files found for Experiment {name}")
            continue

        # Take the first matching file
        file = matching_files[0]
        df = pd.read_csv(file)
        # Group by task_id and check if any attempt was successful
        task_success = df.groupby("task_id")["success"].max()
        accuracy = task_success.mean() * 100  # Convert to percentage
        accuracies[f"{name}"] = accuracy

    # Create bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(
        accuracies.keys(),
        accuracies.values(),
        color="#2E86C1",
        edgecolor="black",
        width=0.6,
    )

    # Customize plot
    plt.title("Zero-shot vs Few-shot Performance Comparison", fontsize=14, pad=20)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for i, (model, accuracy) in enumerate(accuracies.items()):
        plt.text(
            i,
            accuracy + 1,
            f"{accuracy:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Save plot
    plt.savefig(f"{DIR_FIGURES}/{PATH_Q3_FIGURE}", bbox_inches="tight")
    plt.close()


def analyze_few_shot_number_of_examples_results() -> None:
    # Define the experiment names
    experiment_names = [
        "q4-few-shot-1-gemma3:1b",
        "q4-few-shot-3-gemma3:1b",
        "q4-few-shot-5-gemma3:1b",
        "q4-few-shot-7-gemma3:1b",
    ]

    # Get relevant CSV files and store accuracies
    accuracies: Dict[str, float] = {}

    for name in experiment_names:
        pattern = f"{DIR_RESULTS}/{name}*.csv"
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"Warning: No files found for Experiment {name}")
            continue

        # Take the first matching file
        file = matching_files[0]
        df = pd.read_csv(file)
        # Group by task_id and check if any attempt was successful
        task_success = df.groupby("task_id")["success"].max()
        accuracy = task_success.mean() * 100  # Convert to percentage
        accuracies[f"{name}"] = accuracy

    # Create bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(
        accuracies.keys(),
        accuracies.values(),
        color="#2E86C1",
        edgecolor="black",
        width=0.6,
    )

    # Customize plot
    plt.title("Few-shot Number of Examples Performance Comparison", fontsize=14, pad=20)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for i, (model, accuracy) in enumerate(accuracies.items()):
        plt.text(
            i,
            accuracy + 1,
            f"{accuracy:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Save plot
    plt.savefig(f"{DIR_FIGURES}/{PATH_Q4_FIGURE}", bbox_inches="tight")
    plt.close()


def main():
    # Ensure figures directory exists
    os.makedirs(DIR_FIGURES, exist_ok=True)

    analyze_baseline_results()
    analyze_stability()
    analyze_few_shot_improvement_results()
    analyze_few_shot_number_of_examples_results()


if __name__ == "__main__":
    main()
