# Code Generation Agents with MBPP Benchmark

This project explores whether AI Agents equipped with Knowledge Bases created from training datasets can improve code generation quality. We use the Mostly Basic Python Problems (MBPP) benchmark to evaluate different approaches using Gemma3 models.

## Overview

The MBPP benchmark consists of approximately 1,000 crowd-sourced Python programming problems designed for entry-level programmers. Each problem includes:
- Task description
- Code solution
- 3 automated test cases

## Project Goals

We aim to reproduce and improve upon the benchmarks using various approaches:

- [x] Code generation using Gemma3 models
- [x] Safe environment code execution
- [ ] Zero-shot approaches:
  - [x] Basic zero-shot
  - [x] Zero-shot with naive repetition
  - [ ] Zero-shot with smart repetition
- [ ] Few-shot approaches:
  - [x] Basic few-shot
  - [x] Few-shot with repetition
  - Few-shot with loop repetition
- [ ] RAG-based approaches
- [ ] Knowledge Base learning validation

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) - Python package installer and resolver
- [ollama](https://ollama.com/download) - Local large language model runner

### Installation

1. Download required Gemma models:
```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull gemma3:12b
ollama pull gemma3:27b
```

2. Install project dependencies:
```bash
uv pip install .
```

## Usage

### Running Experiments

```bash
# Zero-shot experiments
uv run run_experiments.py --experiment_name zero-shot --model_name gemma3:1b
uv run run_experiments.py --experiment_name zero-shot --model_name gemma3:4b
uv run run_experiments.py --experiment_name zero-shot-naive-repeat --model_name gemma3:1b --num-iterations 3
uv run run_experiments.py --experiment_name zero-shot-self-improving --model_name gemma3:1b --num-iterations 3

# Few-shot experiments
uv run run_experiments.py --experiment_name few-shot --model_name gemma3:1b
uv run run_experiments.py --experiment_name few-shot-naive-repeat --model_name gemma3:1b --num-iterations 3
uv run run_experiments.py --experiment_name few-shot-self-improving --model_name gemma3:1b --num-iterations 3
```

### Evaluation

```bash
uv run run_evaluation.py --results-path results/...
```

### Generate Summary

```bash
uv run run_summary.py
```

## Dataset Details

Based on the [Program Synthesis with Large Language Models](https://arxiv.org/pdf/2108.07732) paper by Austin et al., 2021.

### Dataset Split

- Testing: Task IDs 11-510
- Few-shot prompting: Task IDs 1-10 (not used for training)
- Validation: Task IDs 511-600
- Training: Task IDs 601-974

The original paper used three-shot prompts with task_ids 2, 3, and 4 for few-shot experiments.


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
