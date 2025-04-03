import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Constants
DIR_RESULTS = "results"
DIR_FIGURES = "figures"

# Figure paths
FIGURE_PATHS = {
    "baselines": "Q1_baselines.png",
    "stability": "Q2_stability.png",
    "few_shot_improvement": "Q3_few_shot_improvement.png",
    "few_shot_examples": "Q4_few_shot_number_of_examples.png",
    "reflection_approach": "Q5_reflection_approach.png",
    "reflection_few_shot": "Q6_reflection_few_shot_comparison.png",
    "vector_based_few_shot": "Q7_vector_based_few_shot.png",
    "vector_search_examples": "Q8_vector_search_number_of_examples.png",
}


@dataclass
class ExperimentConfig:
    name: str
    title: str
    figure_path: str
    experiment_names: List[str]


def create_bar_plot(
    accuracies: Dict[str, float],
    title: str,
    xlabel: str,
    ylabel: str = "Accuracy (%)",
    figsize: Tuple[int, int] = (8, 6),
    rotation: int = 0,
) -> None:
    """Create and save a bar plot with the given data and configuration."""
    plt.figure(figsize=figsize)
    plt.bar(
        accuracies.keys(),
        accuracies.values(),
        color="#2E86C1",
        edgecolor="black",
        width=0.6,
    )

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=rotation)
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


def get_experiment_accuracy(experiment_name: str) -> float:
    """Calculate accuracy for a given experiment."""
    pattern = f"{DIR_RESULTS}/{experiment_name}*.csv"
    matching_files = glob.glob(pattern)

    if not matching_files:
        print(f"Warning: No files found for Experiment {experiment_name}")
        return 0.0

    df = pd.read_csv(matching_files[0])
    task_success = df.groupby("task_id")["success"].max()
    return task_success.mean() * 100


def analyze_experiment(config: ExperimentConfig) -> None:
    """Analyze and plot results for a given experiment configuration."""
    accuracies = {
        name: get_experiment_accuracy(name) for name in config.experiment_names
    }

    create_bar_plot(
        accuracies=accuracies, title=config.title, xlabel="Model", rotation=45
    )
    plt.savefig(f"{DIR_FIGURES}/{config.figure_path}", bbox_inches="tight")
    plt.close()


def analyze_stability() -> None:
    """Analyze stability across iterations for the 1B model."""
    pattern = f"{DIR_RESULTS}/q2-gemma3:1b*.csv"
    matching_files = glob.glob(pattern)

    if not matching_files:
        print("Warning: No files found for stability analysis")
        return

    df = pd.read_csv(matching_files[0])
    pivot = pd.pivot_table(
        df, values="success", index="task_id", columns="attempt_number", aggfunc="mean"
    )

    cumulative_accuracy = []
    for col in range(pivot.shape[1]):
        max_success = pivot.iloc[:, : col + 1].max(axis=1)
        accuracy = max_success.mean() * 100
        cumulative_accuracy.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, len(cumulative_accuracy) + 1),
        cumulative_accuracy,
        color="#2E86C1",
        edgecolor="black",
        width=0.6,
    )

    plt.title("Cumulative Accuracy vs. Number of Iterations", fontsize=14, pad=20)
    plt.xlabel("Number of Iterations", fontsize=12)
    plt.ylabel("Cumulative Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, accuracy in enumerate(cumulative_accuracy):
        plt.text(
            i + 1,
            accuracy + 1,
            f"{accuracy:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.savefig(f"{DIR_FIGURES}/{FIGURE_PATHS['stability']}", bbox_inches="tight")
    plt.close()


def main():
    """Main function to run all analyses."""
    os.makedirs(DIR_FIGURES, exist_ok=True)

    # Define experiment configurations
    experiments = [
        ExperimentConfig(
            name="baselines",
            title="Gemma Models Accuracy Comparison",
            figure_path=FIGURE_PATHS["baselines"],
            experiment_names=[
                "q1-zero-shot-gemma3:1b",
                "q1-zero-shot-gemma3:4b",
                "q1-zero-shot-gemma3:12b",
                "q1-zero-shot-gemma3:27b",
            ],
        ),
        ExperimentConfig(
            name="few_shot_improvement",
            title="Zero-shot vs Few-shot Performance Comparison",
            figure_path=FIGURE_PATHS["few_shot_improvement"],
            experiment_names=[
                "q3-zero-shot-gemma3:1b",
                "q3-few-shot-gemma3:1b",
            ],
        ),
        ExperimentConfig(
            name="few_shot_examples",
            title="Few-shot Number of Examples Performance Comparison",
            figure_path=FIGURE_PATHS["few_shot_examples"],
            experiment_names=[
                "q4-few-shot-1-gemma3:1b",
                "q4-few-shot-3-gemma3:1b",
                "q4-few-shot-5-gemma3:1b",
                "q4-few-shot-7-gemma3:1b",
            ],
        ),
        ExperimentConfig(
            name="reflection_approach",
            title="Zero-shot vs Reflection Approach Performance Comparison",
            figure_path=FIGURE_PATHS["reflection_approach"],
            experiment_names=[
                "q5-zero-shot-gemma3:4b",
                "q5-reflection-approach-gemma3:4b",
                "q5-zero-shot-gemma3:12b",
                "q5-reflection-approach-gemma3:12b",
            ],
        ),
        ExperimentConfig(
            name="reflection_few_shot",
            title="Zero-shot vs Few-shot Reflection Approach Performance Comparison",
            figure_path=FIGURE_PATHS["reflection_few_shot"],
            experiment_names=[
                "q5-zero-shot-gemma3:12b",
                "q5-reflection-approach-gemma3:12b",
                "q6-reflection-approach-few-shot-gemma3:12b",
            ],
        ),
        ExperimentConfig(
            name="vector_based_few_shot",
            title="Comparison of Vector Search-based Few-Shot Example Retrieval Against Baselines",
            figure_path=FIGURE_PATHS["vector_based_few_shot"],
            experiment_names=[
                "q3-zero-shot-gemma3:1b",
                "q3-few-shot-gemma3:1b",
                "q7-vector-search-few-shot-gemma3:1b",
            ],
        ),
        ExperimentConfig(
            name="vector_search_examples",
            title="Comparison of Vector Search-based Few-Shot Example Retrieval Against Baselines",
            figure_path=FIGURE_PATHS["vector_search_examples"],
            experiment_names=[
                "q8-vector-search-few-shot-1-gemma3:1b",
                "q8-vector-search-few-shot-3-gemma3:1b",
                "q8-vector-search-few-shot-5-gemma3:1b",
                "q8-vector-search-few-shot-7-gemma3:1b",
            ],
        ),
    ]

    # Run all experiments
    for config in experiments:
        analyze_experiment(config)

    # Run stability analysis separately as it has different plotting logic
    analyze_stability()


if __name__ == "__main__":
    main()
