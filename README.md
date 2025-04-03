# Small Gemma Models for Code Generation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 📋 Overview

This research project investigates whether smaller Gemma models (1B, 4B) can achieve code generation capabilities comparable to their larger counterparts (12B, 27B) through various enhancement techniques. We explore:

- Zero-shot baseline approaches
- Few-shot learning approaches
- Zero- and few-shot approaches with additional reflection from a secondary model
- Vector search-based few-shot example retrieval

## 🔍 Research Questions

1. What are the baseline zero-shot capabilities across different Gemma model sizes?
2. How stable are smaller models' results in zero-shot scenarios?
3. What improvements can be achieved through few-shot approaches?
4. How does the number of examples in few-shot learning influence results?
5. Can additional reflection from another agent regarding the task improve results?
6. Can few-shot examples and additional reflection significantly improve Gemma 12B's results?
7. Can vector search-based few-shot example retrieval improve model performance compared to sequential selection?
8. How does the number of examples affect performance in vector search-based few-shot learning?

## 🧪 Methodology

### Dataset

We utilize a hand-verified subset of the [Mostly Basic Python Problems (MBPP)](https://github.com/google-research/google-research/tree/master/mbpp) benchmark. The original MBPP dataset includes 1,000 Python programming problems for entry-level programmers, each containing:
- Task description
- Reference solution
- 3 test cases for validation

For this study, we used 50 test examples due to computational budget limitations.

### Experimental Approaches

1. **Zero-shot baseline**: Direct code generation without examples
2. **Few-shot baseline**: Direct code generation with examples
3. **Reflection Agent**: Direct code generation with additional task reflection from another agent
4. **Vector Search-based Few-Shot**: Using sentence transformers (all-MiniLM-L6-v2) to compute embeddings and select semantically similar examples based on cosine similarity

### Computational Resources

All experiments were run locally on Apple Silicon M4 Pro processors for reproducibility.

## 📊 Key Findings

### Q1: Zero-Shot Capabilities Across Model Sizes

For our evaluation, we generated one code sample per problem with a temperature setting of 0.5 and assessed whether it passed all three provided test cases. We limited model output tokens to 1000. These settings remained consistent across all subsequent experiments.

![Zero-shot baseline performance](figures/Q1_baselines.png)

Model performance gradually improved with parameter count, starting from ~55% for the 1B model and reaching ~90% for the 27B model.

### Q2: Results Stability in Zero-Shot Scenarios

As we operated on a small sample of data, we evaluated result stability by generating 5 different responses for each query and tracked accumulated accuracy across iterations. Accumulated accuracy measures whether the model had at least one success in N trials, calculated across all test samples. This experiment was performed using Gemma3 1B.

![Zero-shot stability analysis](figures/Q2_stability.png)

Our analysis showed that results stabilized after 2-3 iterations, informing our experimental design choices. For subsequent experiments, we standardized on 3 iterations to ensure result comparability, unless stated otherwise.

### Q3: Zero-Shot vs. Few-Shot Comparison

We analyzed the impact of providing examples in the model prompt by evaluating zero-shot approaches against few-shot approaches with 3 examples. Both experiments were run three times, with accuracy calculated based on whether any of the model trials for each task was successful.

![Zero-Shot vs. Few-Shot Comparison](figures/Q3_few_shot_improvement.png)

We observed slight improvements with the few-shot approach. However, it's worth noting that the zero-shot results were significantly lower than usual (as seen in Q1 and Q2). Increasing iterations or evaluating more examples could produce more reliable results, though both solutions would require greater computational resources. While Q2 suggested results stabilize around 3 iterations, our practical findings indicate this may not always be the case.

### Q4: Impact of Example Count in Few-Shot Learning

We analyzed how the number of examples affects model performance in few-shot scenarios, using a methodology similar to previous research questions (3 runs with accuracy measured by any successful attempt).

![Impact of Example Count in Few-Shot Learning](figures/Q4_few_shot_number_of_examples.png)

We observed that increasing the number of examples in few-shot scenarios significantly improved model performance, showing a clear trend of enhanced accuracy with more examples presented to the model in the prompt.

### Q5: Zero-shot vs. Zero-shot Reflection Approach Comparison

We analyzed whether providing additional task reflection from a Gemma 4B model could improve results. We compared the zero-shot approach with the reflection approach for both 4B and 12B models, running just one iteration due to computational constraints.

![Zero-shot vs Reflection Approach Comparison](figures/Q5_reflection_approach.png)

Surprisingly, the additional reflection component did not strengthen model results, with performance remaining on par for both 4B and 12B models.

### Q6: Zero-shot vs. Few-shot Reflection Approach Comparison

We analyzed whether combining few-shot learning with reflection could improve model performance compared to zero-shot approaches. For this analysis, we used the Gemma 12B model with 3 few-shot examples.

![Zero-shot vs Few-shot Reflection Approach Comparison](figures/Q6_reflection_few_shot_comparison.png)

Unfortunately, the few-shot approach with reflection did not strengthen model results either. This is somewhat understandable given that few-shot approaches didn't consistently improve results in Q3 and Q4.

### Q7: Vector Search-based Few-Shot Example Retrieval

We investigated whether using vector search to select semantically similar examples could improve model performance compared to sequential example selection. We used the sentence transformer model (all-MiniLM-L6-v2) to compute embeddings and find similar examples based on cosine similarity.

![Comparison of Vector Search-based Few-Shot Example Retrieval Against Baselines](figures/Q7_vector_based_few_shot.png)

The results showed that vector search-based example selection provided only a small accuracy improvement.

### Q8: Impact of Example Count in Vector Search-based Few-Shot Learning

Building on Q7, we analyzed how the number of examples selected through vector search affects model performance. We compared different numbers of examples (1, 3, 5, and 7) to understand if more examples lead to better results.

![Impact of Example Count in Vector Search-based Few-Shot Learning](figures/Q8_vector_search_number_of_examples.png)

Increasing the number of examples in vector search-based few-shot learning showed improved accuracy with larger sample sizes.

## Summary

Our research on Gemma models for code generation revealed that larger models (12B, 27B) consistently outperform smaller variants (1B, 4B) in zero-shot tasks. While multiple iterations improved success rates, they plateaued after approximately 3 attempts. Few-shot learning showed modest improvements that scaled with example count, though surprisingly, adding structured reflection components to prompts did not yield significant improvements. Vector-based example selection provided only marginal benefits over sequential selection.

The results suggest that simpler approaches like basic few-shot learning can be as effective as more complex strategies, and that balancing model size with example count may be more practical than pursuing sophisticated prompt engineering techniques.

It's important to note that these conclusions might not be fully valid, as previous experiments showed a lack of result stability.

## 🚀 Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) - Modern Python package installer
- [ollama](https://ollama.com/download) - Local LLM runner
- Python 3.11 or higher

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/small-gemma-models-for-code-generation.git
cd small-gemma-models-for-code-generation
```

2. Download Gemma models:
```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull gemma3:12b
ollama pull gemma3:27b
```

3. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

4. Install dependencies:
```bash
uv pip install .
```

5. (Optional) Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

## 🔧 Usage

### Running Experiments

```bash
# Q1: Basic zero-shot experiments
uv run run_experiments.py --experiment_type single-model --experiment_name q1-zero-shot-gemma3:1b --model_name gemma3:1b
uv run run_experiments.py --experiment_type single-model --experiment_name q1-zero-shot-gemma3:4b --model_name gemma3:4b
uv run run_experiments.py --experiment_type single-model --experiment_name q1-zero-shot-gemma3:12b --model_name gemma3:12b
uv run run_experiments.py --experiment_type single-model --experiment_name q1-zero-shot-gemma3:27b --model_name gemma3:27b

# Q2: Stability analysis
uv run run_experiments.py --experiment_type single-model --experiment_name q2-gemma3:1b --model_name gemma3:1b --num-iterations 5

# Q3: Zero-shot vs. Few-shot Comparison
uv run run_experiments.py --experiment_type single-model --experiment_name q3-zero-shot-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 0
uv run run_experiments.py --experiment_type single-model --experiment_name q3-few-shot-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 3

# Q4: Few-shot Number of Examples Analysis
uv run run_experiments.py --experiment_type single-model --experiment_name q4-few-shot-1-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 1
uv run run_experiments.py --experiment_type single-model --experiment_name q4-few-shot-3-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 3
uv run run_experiments.py --experiment_type single-model --experiment_name q4-few-shot-5-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 5
uv run run_experiments.py --experiment_type single-model --experiment_name q4-few-shot-7-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 7

# Q5: Reflection Approach
uv run run_experiments.py --experiment_type single-model --experiment_name q5-zero-shot-gemma3:4b --model_name gemma3:4b --num-iterations 1 --num-few-shot-examples 0
uv run run_experiments.py --experiment_type single-model --experiment_name q5-zero-shot-gemma3:12b --model_name gemma3:12b --num-iterations 1 --num-few-shot-examples 0
uv run run_experiments.py --experiment_type reflection-approach --experiment_name q5-reflection-approach-gemma3:4b --model_name gemma3:4b --num-iterations 1 --num-few-shot-examples 0
uv run run_experiments.py --experiment_type reflection-approach --experiment_name q5-reflection-approach-gemma3:12b --model_name gemma3:12b --num-iterations 1 --num-few-shot-examples 0

# Q6: Reflection with Few-Shot Examples
uv run run_experiments.py --experiment_type reflection-approach --experiment_name q6-reflection-approach-few-shot-gemma3:12b --model_name gemma3:12b --num-iterations 1 --num-few-shot-examples 2

# Q7: Few-Shot Examples via Vector Search
uv run run_experiments.py --experiment_type single-model --experiment_name q7-vector-search-few-shot-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 3 --use-vector-search

# Q8: Few-Shot Examples via Vector Search Number of Examples Analysis
uv run run_experiments.py --experiment_type single-model --experiment_name q8-vector-search-few-shot-1-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 1 --use-vector-search
uv run run_experiments.py --experiment_type single-model --experiment_name q8-vector-search-few-shot-3-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 3 --use-vector-search
uv run run_experiments.py --experiment_type single-model --experiment_name q8-vector-search-few-shot-5-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 5 --use-vector-search
uv run run_experiments.py --experiment_type single-model --experiment_name q8-vector-search-few-shot-7-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 7 --use-vector-search
```

### Evaluation

Run evaluation for specific results:
```bash
# Q1 Experiments
uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:1b*.json
uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:4b*.json
uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:12b*.json
uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:27b*.json

# Q2 Experiments
uv run run_evaluation.py --results-path results/q2-gemma3:1b*.json

# Q3 Experiments
uv run run_evaluation.py --results-path results/q3-zero-shot-gemma3:1b*.json
uv run run_evaluation.py --results-path results/q3-few-shot-gemma3:1b*.json

# Q4 Experiments
uv run run_evaluation.py --results-path results/q4-few-shot-1-gemma3:1b*.json
uv run run_evaluation.py --results-path results/q4-few-shot-3-gemma3:1b*.json
uv run run_evaluation.py --results-path results/q4-few-shot-5-gemma3:1b*.json
uv run run_evaluation.py --results-path results/q4-few-shot-7-gemma3:1b*.json

# Q5 Experiments
uv run run_evaluation.py --results-path results/q5-zero-shot-gemma3:4b*.json
uv run run_evaluation.py --results-path results/q5-zero-shot-gemma3:12b*.json
uv run run_evaluation.py --results-path results/q5-reflection-approach-gemma3:4b*.json
uv run run_evaluation.py --results-path results/q5-reflection-approach-gemma3:12b*.json

# Q6 Experiments
uv run run_evaluation.py --results-path results/q6-reflection-approach-few-shot-gemma3:12b*.json

# Q7 Experiments
uv run run_evaluation.py --results-path results/q7-vector-search-few-shot-gemma3:1b*.json

# Q8 Experiments
uv run run_evaluation.py --results-path results/q8-vector-search-few-shot-1-gemma3:1b*.json
uv run run_evaluation.py --results-path results/q8-vector-search-few-shot-3-gemma3:1b*.json
uv run run_evaluation.py --results-path results/q8-vector-search-few-shot-5-gemma3:1b*.json
uv run run_evaluation.py --results-path results/q8-vector-search-few-shot-7-gemma3:1b*.json
```

Generate visualizations:
```bash
uv run run_analysis.py
```

## 📝 License

[Apache License 2.0](LICENSE)
