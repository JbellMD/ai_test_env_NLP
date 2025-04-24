#!/usr/bin/env python
"""
Debug script to check the structure of the IMDB dataset from Hugging Face.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset

def debug_dataset_structure():
    """
    Load and analyze the IMDB dataset structure.
    """
    print("Loading IMDB dataset...")
    try:
        dataset = load_dataset("imdb")
        print(f"Dataset loaded successfully. Type: {type(dataset)}")
        
        # Print dataset splits
        print(f"\nDataset splits: {list(dataset.keys())}")
        
        # Check train split structure
        print("\nTrain split structure:")
        train_features = dataset["train"].features
        print(f"Features: {train_features}")
        
        # Check a sample
        print("\nFirst training example:")
        sample = dataset["train"][0]
        print(f"Type: {type(sample)}")
        print(f"Sample: {sample}")
        
        # Check each field
        print("\nIterating through fields:")
        for key, value in sample.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, str):
                print(f"    First 50 chars: {value[:50]}...")
            else:
                print(f"    Value: {value}")
        
        # Try the notebook's approach
        print("\nTrying notebook's approach:")
        try:
            print(f"Text: {sample['text'][:50]}...")
        except Exception as e:
            print(f"Error accessing 'text': {type(e).__name__}: {str(e)}")
        
    except Exception as e:
        print(f"Error loading dataset: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    debug_dataset_structure()
