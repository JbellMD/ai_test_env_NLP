#!/usr/bin/env python
"""
Script to create a prediction-only version of the text classification notebook.
"""

import os
import json
from pathlib import Path

# Get project root
project_root = Path(__file__).resolve().parent.parent

def create_prediction_only_notebook():
    """
    Create a prediction-only version of the text classification notebook.
    """
    target_notebook_path = os.path.join(project_root, "notebooks", "01_text_classification_prediction_only.ipynb")
    
    print(f"Creating prediction-only version of text classification notebook...")
    
    # Create a new notebook from scratch with minimal cells
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
            "# Text Classification Prediction Demo\n",
            "\n",
            "This notebook demonstrates the text classification inference capabilities of the NLP toolkit, without requiring model training."
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
            "from src.models.classifier import TransformerClassifier\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np"
        ]
    })
    
    # Add dataset sample creation cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Sample texts for demonstration\n",
            "sample_texts = [\n",
            "    \"I loved this movie! The plot was amazing and the characters were so well developed.\",\n",
            "    \"This was a terrible waste of time. The story made no sense and the actors seemed bored.\",\n",
            "    \"A decent film but nothing special. I'd watch it again if there's nothing else to see.\",\n",
            "    \"The movie was visually stunning but the dialogue was a bit weak.\",\n",
            "    \"One of the worst movies I've ever seen. Complete disaster from start to finish.\"\n",
            "]\n",
            "\n",
            "# We'll use these for demonstration without needing to load a real dataset\n",
            "print(f\"Number of sample texts: {len(sample_texts)}\")\n",
            "print(f\"\\nExample: {sample_texts[0]}\")"
        ]
    })
    
    # Add model loading and configuration
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Model configuration\n",
            "MODEL_NAME = \"distilbert-base-uncased\"  # Use a small model for faster loading\n",
            "TASK = \"classification\"  # Binary text classification\n",
            "\n",
            "# Initialize the classifier\n",
            "classifier = TransformerClassifier(\n",
            "    model_name=MODEL_NAME,\n",
            "    num_labels=2,  # Binary classification for sentiment\n",
            "    problem_type=\"single_label_classification\"\n",
            ")\n",
            "\n",
            "# Display model information\n",
            "print(f\"Model: {MODEL_NAME}\")\n",
            "print(f\"Task: {TASK}\")\n",
            "print(f\"Number of parameters: {classifier.get_model_size():,}\")"
        ]
    })
    
    # Add prediction cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Make predictions on sample texts\n",
            "predictions = classifier.predict(sample_texts)\n",
            "\n",
            "# Display predictions\n",
            "for i, (text, pred) in enumerate(zip(sample_texts, predictions)):\n",
            "    sentiment = \"Positive\" if pred == 1 else \"Negative\"\n",
            "    print(f\"Text {i+1}: {sentiment}\")\n",
            "    print(f\"   {text[:50]}...\")\n",
            "    print()"
        ]
    })
    
    # Add visualization cell
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Get raw prediction scores (will include probabilities for both classes)\n",
            "# For this demonstration we'll generate mock probabilities\n",
            "# In a real application, you'd use classifier.predict_proba(sample_texts)\n",
            "mock_probabilities = [\n",
            "    [0.15, 0.85],  # High positive probability\n",
            "    [0.92, 0.08],  # High negative probability\n",
            "    [0.45, 0.55],  # Slightly positive\n",
            "    [0.40, 0.60],  # Moderately positive\n",
            "    [0.85, 0.15]   # High negative probability\n",
            "]\n",
            "\n",
            "# Visualize the classification confidence for each sample\n",
            "plt.figure(figsize=(10, 6))\n",
            "x = np.arange(len(sample_texts))\n",
            "width = 0.35\n",
            "\n",
            "plt.bar(x - width/2, [p[0] for p in mock_probabilities], width, label='Negative')\n",
            "plt.bar(x + width/2, [p[1] for p in mock_probabilities], width, label='Positive')\n",
            "\n",
            "plt.ylabel('Probability')\n",
            "plt.title('Classification Confidence')\n",
            "plt.xticks(x, [f'Text {i+1}' for i in range(len(sample_texts))])\n",
            "plt.legend()\n",
            "plt.ylim(0, 1)\n",
            "\n",
            "for i, v in enumerate([p[0] for p in mock_probabilities]):\n",
            "    plt.text(i - width/2, v + 0.05, f'{v:.2f}', ha='center')\n",
            "    \n",
            "for i, v in enumerate([p[1] for p in mock_probabilities]):\n",
            "    plt.text(i + width/2, v + 0.05, f'{v:.2f}', ha='center')\n",
            "    \n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # Add a conclusion cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusion\n",
            "\n",
            "This notebook demonstrates how to use the TransformerClassifier for text classification inference. The model works with pretrained weights from Hugging Face and can be used for a variety of text classification tasks including:\n",
            "\n",
            "- Sentiment analysis\n",
            "- Topic classification\n",
            "- Intent detection\n",
            "- Content categorization\n",
            "\n",
            "To train a model on your own data, refer to the more comprehensive notebooks and examples in the toolkit."
        ]
    })
    
    # Save the new notebook
    with open(target_notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Saved prediction-only notebook to: {target_notebook_path}")

if __name__ == "__main__":
    create_prediction_only_notebook()
