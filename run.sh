#!/bin/bash

echo "🚀 Starting experiments..."

# echo -e "\n📊 Running Q1 Zero-shot Experiments"
# echo "====================================="

# echo -e "\n🔍 Running Gemma 3 1B model..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q1-zero-shot-gemma3:1b --model_name gemma3:1b --num-few-shot-examples 0

# echo -e "\n🔍 Running Gemma 3 4B model..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q1-zero-shot-gemma3:4b --model_name gemma3:4b --num-few-shot-examples 0

# echo -e "\n🔍 Running Gemma 3 12B model..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q1-zero-shot-gemma3:12b --model_name gemma3:12b --num-few-shot-examples 0

# echo -e "\n🔍 Running Gemma 3 27B model..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q1-zero-shot-gemma3:27b --model_name gemma3:27b --num-few-shot-examples 0

# echo -e "\n📊 Running Q2 Experiment"
# echo "=========================="
# echo -e "\n🔍 Running Gemma 3 1B model with 5 iterations..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q2-gemma3:1b --model_name gemma3:1b --num-iterations 5 --num-few-shot-examples 0

# echo -e "\n📊 Running Q3 Experiments"
# echo "=========================="
# echo -e "\n🔍 Running Q3 Zero-shot Gemma 3 1B model..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q3-zero-shot-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 0

# echo -e "\n🔍 Running Q3 Few-shot Gemma 3 1B model..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q3-few-shot-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 3

# echo -e "\n📊 Running Q4 Experiments"
# echo "=========================="
# echo -e "\n🔍 Running Q4 Few-shot Gemma 3 1B model with 1 example..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q4-few-shot-1-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 1

# echo -e "\n🔍 Running Q4 Few-shot Gemma 3 1B model with 3 examples..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q4-few-shot-3-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 3

# echo -e "\n🔍 Running Q4 Few-shot Gemma 3 1B model with 5 examples..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q4-few-shot-5-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 5

# echo -e "\n🔍 Running Q4 Few-shot Gemma 3 1B model with 7 examples..."
# uv run run_experiments.py --experiment_type single-model --experiment_name q4-few-shot-7-gemma3:1b --model_name gemma3:1b --num-iterations 3 --num-few-shot-examples 7

echo -e "\n📊 Running Q5 Experiments"
echo "=========================="
echo -e "\n🔍 Running Q5 Zero-shot Gemma 3 12B model..."
uv run run_experiments.py --experiment_type single-model --experiment_name q5-zero-shot-gemma3:12b --model_name gemma3:12b --num-iterations 1 --num-few-shot-examples 0

echo -e "\n🔍 Running Q5 Reflection Approach with Gemma 3 12B model..."
uv run run_experiments.py --experiment_type reflection-approach --experiment_name q5-reflection-approach-gemma3:12b --model_name gemma3:12b --num-iterations 1 --num-few-shot-examples 0

echo -e "\n📈 Running Evaluations"
echo "======================"

# echo -e "\n📊 Evaluating Q1 Zero-shot Gemma 3 1B results..."
# uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:1b*.json

# echo -e "\n📊 Evaluating Q1 Zero-shot Gemma 3 4B results..."
# uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:4b*.json

# echo -e "\n📊 Evaluating Q1 Zero-shot Gemma 3 12B results..."
# uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:12b*.json

# echo -e "\n📊 Evaluating Q1 Zero-shot Gemma 3 27B results..."
# uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:27b*.json

# echo -e "\n📊 Evaluating Q2 Gemma 3 1B results..."
# uv run run_evaluation.py --results-path results/q2-gemma3:1b*.json

# echo -e "\n📊 Evaluating Q3 Zero-shot Gemma 3 1B results..."
# uv run run_evaluation.py --results-path results/q3-zero-shot-gemma3:1b*.json

# echo -e "\n📊 Evaluating Q3 Few-shot Gemma 3 1B results..."
# uv run run_evaluation.py --results-path results/q3-few-shot-gemma3:1b*.json

# echo -e "\n📊 Evaluating Q4 Few-shot Gemma 3 1B results..."
# uv run run_evaluation.py --results-path results/q4-few-shot-1-gemma3:1b*.json

# echo -e "\n📊 Evaluating Q4 Few-shot Gemma 3 1B results..."
# uv run run_evaluation.py --results-path results/q4-few-shot-3-gemma3:1b*.json

# echo -e "\n📊 Evaluating Q4 Few-shot Gemma 3 1B results..."
# uv run run_evaluation.py --results-path results/q4-few-shot-5-gemma3:1b*.json

# echo -e "\n📊 Evaluating Q4 Few-shot Gemma 3 1B results..."
# uv run run_evaluation.py --results-path results/q4-few-shot-7-gemma3:1b*.json

echo -e "\n📊 Evaluating Q5 Zero-shot Gemma 3 12B results..."
uv run run_evaluation.py --results-path results/q5-zero-shot-gemma3:12b*.json

echo -e "\n📊 Evaluating Q5 Reflection Approach with Gemma 3 12B results..."
uv run run_evaluation.py --results-path results/q5-reflection-approach-gemma3:12b*.json

echo -e "\n📊 Running Final Analysis"
echo "========================"
uv run run_analysis.py

echo -e "\n✨ All experiments completed!"
