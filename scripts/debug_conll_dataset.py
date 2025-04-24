#!/usr/bin/env python
"""
Debug script to understand the structure of the ConLL2003 dataset.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset

def debug_conll_dataset():
    """
    Load and analyze the ConLL-2003 dataset structure.
    """
    print("Loading ConLL-2003 dataset...")
    try:
        dataset = load_dataset("conll2003", trust_remote_code=True)
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
        
        # Check each field
        print("\nIterating through fields:")
        for key, value in sample.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, list):
                print(f"    First 10 elements: {value[:10]}")
                
        # Try the notebook's approach
        print("\nTrying notebook's approach:")
        try:
            print(f"Tokens: {sample['tokens'][:10]}...")
            print(f"NER tags: {sample['ner_tags'][:10]}...")
        except Exception as e:
            print(f"Error accessing fields: {type(e).__name__}: {str(e)}")
            
        # Check direct iteration 
        print("\nDirect iteration over example:")
        try:
            for idx, item in enumerate(sample):
                if idx < 5:  # Only show first 5
                    print(f"  Item {idx}: {item}")
        except Exception as e:
            print(f"Error iterating: {type(e).__name__}: {str(e)}")
        
        # Get example format
        print("\nPrinting raw example format:")
        print(f"{sample}")
        
    except Exception as e:
        print(f"Error loading dataset: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    debug_conll_dataset()
