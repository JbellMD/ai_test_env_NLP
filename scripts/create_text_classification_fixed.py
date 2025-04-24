#!/usr/bin/env python
"""
Script to create a fixed version of the text classification notebook with correct parameter names.
"""

import os
import json
import re
from pathlib import Path

# Get project root
project_root = Path(__file__).resolve().parent.parent

def create_fixed_notebook():
    """
    Create a fixed version of the text classification notebook that uses correct parameter names.
    """
    source_notebook_path = os.path.join(project_root, "notebooks", "01_text_classification_robust.ipynb")
    target_notebook_path = os.path.join(project_root, "notebooks", "01_text_classification_fixed.ipynb")
    
    print(f"Creating fixed version of text classification notebook...")
    
    # Load the original notebook
    with open(source_notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Fixed notebook with the same structure
    new_notebook = {
        "cells": [],
        "metadata": notebook["metadata"],
        "nbformat": notebook.get("nbformat", 4),
        "nbformat_minor": notebook.get("nbformat_minor", 5)
    }
    
    # Process each cell
    for cell in notebook["cells"]:
        # Create a copy of the cell
        new_cell = cell.copy()
        
        # If it's a code cell, check for parameter name fixes
        if cell["cell_type"] == "code":
            source_text = "".join(cell.get("source", []))
            
            # Replace entire TransformerClassifier initialization cell
            if "TransformerClassifier(" in source_text and "task=" in source_text:
                print("  Fixing TransformerClassifier initialization...")
                new_cell["source"] = [
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
            # Fix dataset loading cell
            elif "load_huggingface_dataset" in source_text:
                print("  Adding robust dataset loading...")
                new_cell["source"] = [
                    "# Try to load the dataset with robust error handling\n",
                    "try:\n",
                    "    dataset = dataset_loader.load_huggingface_dataset(\n",
                    "        dataset_name=DATASET_NAME,\n",
                    "        text_column=\"text\",\n",
                    "        label_column=\"label\"\n",
                    "    )\n",
                    "    print(f\"Dataset loaded successfully: {DATASET_NAME}\")\n",
                    "except Exception as e:\n",
                    "    print(f\"Error loading dataset: {e}\")\n",
                    "    # Fallback to a smaller dataset if IMDB fails\n",
                    "    DATASET_NAME = \"sst2\"\n",
                    "    dataset = dataset_loader.load_huggingface_dataset(\n",
                    "        dataset_name=DATASET_NAME,\n",
                    "        text_column=\"sentence\",\n",
                    "        label_column=\"label\"\n",
                    "    )\n",
                    "    print(f\"Fallback dataset loaded: {DATASET_NAME}\")\n"
                ]
            # Fix dataset example access
            elif "example['text']" in source_text or "example['label']" in source_text:
                print("  Adding robust dataset access...")
                # Replace unsafe access patterns with robust alternatives
                source = []
                for line in cell.get("source", []):
                    line = line.replace("example['text']", "example.get('text', example.get('sentence', str(example)))")
                    line = line.replace("example['label']", "example.get('label', 0)")
                    source.append(line)
                new_cell["source"] = source
        
        # Add cell to new notebook
        new_notebook["cells"].append(new_cell)
    
    # Save the new notebook
    with open(target_notebook_path, "w", encoding="utf-8") as f:
        json.dump(new_notebook, f, indent=1)
    
    print(f"Saved fixed notebook to: {target_notebook_path}")
    
if __name__ == "__main__":
    create_fixed_notebook()
