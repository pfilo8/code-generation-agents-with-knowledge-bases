import pandas as pd
import matplotlib.pyplot as plt
import glob

# Get all CSV files from results directory
csv_files = glob.glob("results/*.csv")

# Store results
model_accuracies = {}


# Process each CSV file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)

    # Calculate accuracy (mean of success column)
    accuracy = df["success"].mean()

    # Extract model name and experiment name from the file
    model_name = df["model"].iloc[0]
    experiment_name = df["experiment_name"].iloc[0]

    # Create a combined key for plotting
    plot_key = f"{model_name}\n({experiment_name})"

    # Store results
    model_accuracies[plot_key] = accuracy * 100  # Convert to percentage

# Create bar plot
plt.figure(figsize=(14, 7))  # Increased width for longer labels
plt.bar(
    model_accuracies.keys(),
    model_accuracies.values(),
    color="#2E86C1",  # Nice blue color
    edgecolor="black",  # Black edges
    width=0.6,
)  # Slightly narrower bars

# Customize plot
plt.title("Model Accuracy Comparison", fontsize=14, pad=20)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)

# Set y-axis limits from 0 to 100
plt.ylim(0, 100)

# Add grid for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add value labels on top of bars
for i, (model_exp, accuracy) in enumerate(model_accuracies.items()):
    model, exp = model_exp.split("\n")
    plt.text(
        i,
        accuracy + 1,
        f"{accuracy:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Adjust layout with more bottom margin for longer labels
plt.subplots_adjust(bottom=0.2)

# Save plot
plt.savefig("model_accuracies.png", bbox_inches="tight")
plt.close()

# Print results
print("\nResults by Model and Experiment:")
print("-" * 40)
for model_exp, accuracy in model_accuracies.items():
    model, exp = model_exp.split("\n")
    print(f"{model} - {exp}: {accuracy:.1f}%")
