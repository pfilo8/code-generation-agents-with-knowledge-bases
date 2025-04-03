#!/bin/bash

echo "🚀 Starting experiments..."

echo -e "\n📊 Running Q1 Zero-shot Experiments"
echo "====================================="

echo -e "\n🔍 Running Gemma 3 1B model..."
uv run run_experiments.py --experiment_type zero-shot --experiment_name q1-zero-shot-gemma3:1b --model_name gemma3:1b

echo -e "\n🔍 Running Gemma 3 4B model..."
uv run run_experiments.py --experiment_type zero-shot --experiment_name q1-zero-shot-gemma3:4b --model_name gemma3:4b

echo -e "\n🔍 Running Gemma 3 12B model..."
uv run run_experiments.py --experiment_type zero-shot --experiment_name q1-zero-shot-gemma3:12b --model_name gemma3:12b

echo -e "\n🔍 Running Gemma 3 27B model..."
uv run run_experiments.py --experiment_type zero-shot --experiment_name q1-zero-shot-gemma3:27b --model_name gemma3:27b

echo -e "\n📊 Running Q2 Experiment"
echo "=========================="
echo -e "\n🔍 Running Gemma 3 1B model with 5 iterations..."
uv run run_experiments.py --experiment_type zero-shot --experiment_name q2-gemma3:1b --model_name gemma3:1b --num-iterations 5

echo -e "\n📈 Running Evaluations"
echo "======================"

echo -e "\n📊 Evaluating Q1 Zero-shot Gemma 3 1B results..."
uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:1b*.json

echo -e "\n📊 Evaluating Q1 Zero-shot Gemma 3 4B results..."
uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:4b*.json

echo -e "\n📊 Evaluating Q1 Zero-shot Gemma 3 12B results..."
uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:12b*.json

echo -e "\n📊 Evaluating Q1 Zero-shot Gemma 3 27B results..."
uv run run_evaluation.py --results-path results/q1-zero-shot-gemma3:27b*.json

echo -e "\n📊 Evaluating Q2 Gemma 3 1B results..."
uv run run_evaluation.py --results-path results/q2-gemma3:1b*.json

echo -e "\n📊 Running Final Analysis"
echo "========================"
uv run run_analysis.py

echo -e "\n✨ All experiments completed!"
