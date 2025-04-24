#!/usr/bin/env python
"""
Script to create a prediction-only version of the NER notebook that skips the problematic training sections.
"""

import os
import json
from pathlib import Path

# Get project root
project_root = Path(__file__).resolve().parent.parent

def create_prediction_only_notebook():
    """
    Create a simplified version of the NER notebook that focuses on prediction functionality.
    This avoids the training section with tensor dimension issues.
    """
    source_notebook_path = os.path.join(project_root, "notebooks", "02_named_entity_recognition_demo.ipynb")
    target_notebook_path = os.path.join(project_root, "notebooks", "02_named_entity_recognition_prediction_only.ipynb")
    
    print(f"Creating prediction-only version of NER notebook...")
    
    # Load the original notebook
    with open(source_notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Create a new notebook with selected cells
    new_notebook = {
        "cells": [],
        "metadata": notebook["metadata"],
        "nbformat": notebook.get("nbformat", 4),
        "nbformat_minor": notebook.get("nbformat_minor", 5)
    }
    
    # Helper function to add markdown section dividers
    def add_section_divider(title):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {title}"]
        }
    
    # Add introduction and setup cells
    for i, cell in enumerate(notebook["cells"]):
        # Include intro, imports, and configuration cells
        if i <= 6:  # Import cells and configuration
            new_notebook["cells"].append(cell)
    
    # Add a note about this being a modified version
    note_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Note: Prediction-Only Version\n\n",
            "This is a simplified version of the NER demo notebook that focuses on prediction rather than training.",
            "It demonstrates how to use pre-trained NER models for inference, but skips the training sections",
            "which require specific dataset formatting and tensor shapes."
        ]
    }
    new_notebook["cells"].append(note_cell)
    
    # Add the pre-trained model cells
    new_notebook["cells"].append(add_section_divider("Using Pre-trained NER Models"))
    
    # Find and add model loading and prediction cells 
    for i, cell in enumerate(notebook["cells"]):
        # Model loading and sample predictions
        if "NERModel(model_name=" in str(cell.get("source", [])) or "sample_texts" in str(cell.get("source", [])):
            new_notebook["cells"].append(cell)
            
        # Visualization code
        if "visualize_ner_prediction" in str(cell.get("source", [])):
            new_notebook["cells"].append(cell)
    
    # Add custom prediction examples
    custom_examples_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Try with your own custom text examples\n",
            "custom_texts = [\n",
            "    \"Google and Facebook announced new AI research partnerships at Stanford University yesterday.\",\n",
            "    \"Tesla CEO Elon Musk visited their Berlin Gigafactory in Germany last month.\",\n",
            "    \"The World Health Organization released new guidelines for COVID-19 prevention in New York.\"\n",
            "]\n",
            "\n",
            "# Run predictions\n",
            "custom_predictions = ner_model.predict(custom_texts)\n",
            "\n",
            "# Display results\n",
            "for i, (text, entities) in enumerate(zip(custom_texts, custom_predictions)):\n",
            "    print(f\"\\nText {i+1}: {text}\")\n",
            "    print(\"Entities:\")\n",
            "    for entity in entities:\n",
            "        entity_text = text[entity['start']:entity['end']]\n",
            "        entity_type = entity['entity'].replace('B-', '').replace('I-', '') if '-' in entity['entity'] else entity['entity']\n",
            "        print(f\"  {entity_text} ({entity_type}): {entity['score']:.3f}\")\n",
            "\n",
            "# Visualize the first custom example\n",
            "visualize_ner_prediction(custom_texts[0], custom_predictions[0])"
        ],
        "execution_count": None,
        "outputs": []
    }
    new_notebook["cells"].append(add_section_divider("Custom Examples"))
    new_notebook["cells"].append(custom_examples_cell)
    
    # Save the new notebook
    with open(target_notebook_path, "w", encoding="utf-8") as f:
        json.dump(new_notebook, f, indent=1)
    
    print(f"Saved prediction-only notebook to: {target_notebook_path}")
    
if __name__ == "__main__":
    create_prediction_only_notebook()
