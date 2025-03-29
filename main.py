import json
import pathlib

import ollama


def load_data(filename: pathlib.Path) -> dict:
    # Load the first json file found
    with open(filename) as f:
        return json.load(f)


def generate_response(prompt: str, model_name: str = "gemma3:27b") -> str:
    """Generate a response using Gemma 3B model via Ollama."""
    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        return response["response"]
    except Exception as e:
        return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    mbpp_filepath = pathlib.Path("data/sanitized-mbpp.json")
    data = load_data(mbpp_filepath)

    model_name = "gemma3:12b"
    example = data[1]
    prompt = example["prompt"]
    print(prompt)

    response = generate_response(prompt=prompt, model_name=model_name)
    print(response)
