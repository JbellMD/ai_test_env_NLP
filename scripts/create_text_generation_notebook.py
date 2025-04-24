#!/usr/bin/env python
"""
Script to create a proper text generation notebook.
"""

import os
import json
from pathlib import Path

# Get project root
project_root = Path(__file__).resolve().parent.parent

def create_text_generation_notebook():
    """
    Create a proper text generation notebook that demonstrates language model capabilities.
    """
    target_notebook_path = os.path.join(project_root, "notebooks", "04_text_generation_demo.ipynb")
    
    print(f"Creating text generation notebook...")
    
    # Create a new notebook from scratch
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Add title and introduction
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Text Generation Demo\n",
            "\n",
            "This notebook demonstrates text generation capabilities using language models from the NLP toolkit."
        ]
    })
    
    # Add imports and setup
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Setup path to allow importing from the src directory\n",
            "import sys\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "# Add the project root to the Python path\n",
            "project_root = str(Path().resolve().parent)\n",
            "if project_root not in sys.path:\n",
            "    sys.path.append(project_root)\n",
            "\n",
            "# Import necessary modules\n",
            "from transformers import pipeline\n",
            "import torch\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np"
        ]
    })
    
    # Add model configuration
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Model configuration\n",
            "MODEL_NAME = \"gpt2\"  # Small model for faster loading and generation\n",
            "MAX_LENGTH = 50  # Maximum length of generated text\n",
            "\n",
            "print(f\"Using model: {MODEL_NAME}\")\n",
            "print(f\"Max generation length: {MAX_LENGTH}\")"
        ]
    })
    
    # Add base generation with Hugging Face pipeline
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Create text generation pipeline\n",
            "generator = pipeline('text-generation', model=MODEL_NAME)\n",
            "\n",
            "# Sample prompt\n",
            "prompt = \"In the future, artificial intelligence will\"\n",
            "\n",
            "# Generate text\n",
            "generated_text = generator(\n",
            "    prompt, \n",
            "    max_length=MAX_LENGTH, \n",
            "    num_return_sequences=1,\n",
            "    pad_token_id=generator.tokenizer.eos_token_id\n",
            ")\n",
            "\n",
            "# Display results\n",
            "print(f\"Prompt: {prompt}\")\n",
            "print(\"\\nGenerated text:\")\n",
            "print(generated_text[0]['generated_text'])"
        ]
    })
    
    # Add multiple prompts
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Multiple prompts for comparison\n",
            "prompts = [\n",
            "    \"The best way to learn programming is\",\n",
            "    \"Climate change will impact the world by\",\n",
            "    \"The future of natural language processing includes\"\n",
            "]\n",
            "\n",
            "# Generate text for each prompt\n",
            "for i, prompt in enumerate(prompts):\n",
            "    print(f\"\\nPrompt {i+1}: {prompt}\")\n",
            "    \n",
            "    # Generate with shorter length for multiple examples\n",
            "    result = generator(\n",
            "        prompt, \n",
            "        max_length=30, \n",
            "        num_return_sequences=1,\n",
            "        pad_token_id=generator.tokenizer.eos_token_id\n",
            "    )\n",
            "    \n",
            "    print(\"Generated text:\")\n",
            "    print(result[0]['generated_text'])\n",
            "    print(\"-\" * 50)"
        ]
    })
    
    # Add generation with parameters
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Exploring generation parameters\n",
            "prompt = \"Artificial intelligence will revolutionize\"\n",
            "\n",
            "# Different temperature values\n",
            "temperatures = [0.7, 1.0, 1.5]\n",
            "\n",
            "for temp in temperatures:\n",
            "    print(f\"\\nTemperature: {temp}\")\n",
            "    \n",
            "    result = generator(\n",
            "        prompt, \n",
            "        max_length=40, \n",
            "        temperature=temp,\n",
            "        num_return_sequences=1,\n",
            "        pad_token_id=generator.tokenizer.eos_token_id\n",
            "    )\n",
            "    \n",
            "    print(result[0]['generated_text'])\n",
            "    print(\"-\" * 50)"
        ]
    })
    
    # Add a conclusion cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusion\n",
            "\n",
            "This notebook demonstrates basic text generation capabilities using transformers-based language models. The generation quality can be controlled through parameters like:\n",
            "\n",
            "- **Temperature**: Controls randomness (higher = more random)\n",
            "- **Top-k**: Limits vocabulary to top k most likely tokens\n",
            "- **Top-p (nucleus sampling)**: Samples from the smallest set of tokens whose cumulative probability exceeds p\n",
            "- **Repetition penalty**: Reduces repetitive text\n",
            "\n",
            "For more advanced generation capabilities, the full API provides additional control options and model choices."
        ]
    })
    
    # Save the new notebook
    with open(target_notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Saved text generation notebook to: {target_notebook_path}")

if __name__ == "__main__":
    create_text_generation_notebook()
