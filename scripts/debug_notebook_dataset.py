#!/usr/bin/env python
"""
Debug script to understand how the notebook is handling datasets.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from transformers import AutoTokenizer
from src.data.preprocessing import TextPreprocessor
from src.data.data_loader import get_text_classification_loader

# Configuration
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"
MAX_LENGTH = 128

def print_separator():
    print("\n" + "=" * 50 + "\n")

def debug_dataset_handling():
    """Debug how the notebook handles datasets."""
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    preprocessor = TextPreprocessor()
    
    # Original HuggingFace dataset
    print("Loading original dataset...")
    original_dataset = load_dataset(DATASET_NAME)
    
    print_separator()
    print("ORIGINAL DATASET STRUCTURE")
    print(f"Type: {type(original_dataset)}")
    print(f"Keys: {original_dataset.keys()}")
    print(f"Train examples: {len(original_dataset['train'])}")
    
    # Check first example
    print_separator()
    print("ORIGINAL DATASET FIRST EXAMPLE")
    first_example = original_dataset["train"][0]
    print(f"Type: {type(first_example)}")
    print(f"Keys: {first_example.keys()}")
    print(f"Text (first 50 chars): {first_example['text'][:50]}...")
    print(f"Label: {first_example['label']}")
    
    # Now try with our custom loader
    print_separator()
    print("LOADING WITH CUSTOM LOADER")
    dataset_loader = get_text_classification_loader(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_length=MAX_LENGTH
    )
    
    custom_dataset = dataset_loader.load_huggingface_dataset(
        dataset_name=DATASET_NAME,
        text_column="text",
        label_column="label"
    )
    
    print_separator()
    print("CUSTOM DATASET STRUCTURE")
    print(f"Type: {type(custom_dataset)}")
    print(f"Keys: {custom_dataset.keys()}")
    for split in custom_dataset.keys():
        print(f"Split '{split}' type: {type(custom_dataset[split])}")
        print(f"Split '{split}' length: {len(custom_dataset[split])}")
    
    # Check first example with careful error handling
    print_separator()
    print("CUSTOM DATASET FIRST EXAMPLE")
    try:
        first_custom_example = custom_dataset["train"][0]
        print(f"Type: {type(first_custom_example)}")
        
        if hasattr(first_custom_example, 'keys'):
            print(f"Keys: {first_custom_example.keys()}")
        elif isinstance(first_custom_example, dict):
            print(f"Dict keys: {first_custom_example.keys()}")
        else:
            print(f"Not a dictionary-like object. Raw value: {repr(first_custom_example)}")
            
        # Try to access 'text' field
        try:
            if isinstance(first_custom_example, dict) and 'text' in first_custom_example:
                print(f"Text: {first_custom_example['text'][:50]}...")
            else:
                print("No 'text' field found or not accessible as dictionary")
                
            # Try string access
            try:
                text_as_string = str(first_custom_example)
                print(f"As string: {text_as_string[:50]}...")
            except Exception as e:
                print(f"Cannot convert to string: {str(e)}")
                
        except Exception as e:
            print(f"Error accessing 'text': {type(e).__name__}: {str(e)}")
            
    except Exception as e:
        print(f"Error accessing first example: {type(e).__name__}: {str(e)}")
    
    # Try to fix the issue - directly apply the fix to our data loader
    print_separator()
    print("ATTEMPTING TO FIX THE DATASET LOADER")
    
    try:
        # Create a fix for the data loader's __getitem__ method
        from src.data.data_loader import CustomNLPDataset
        
        # Store original method for comparison
        original_getitem = CustomNLPDataset.__getitem__
        
        # Define a patched version
        def patched_getitem(self, idx):
            # Get the text and maybe process it
            text = self.texts[idx]
            
            # Return a properly formatted dictionary
            result = {"text": text}
            
            # Add label if available
            if self.labels is not None:
                result["label"] = self.labels[idx]
                
            return result
        
        # Apply the patch
        CustomNLPDataset.__getitem__ = patched_getitem
        
        # Test with the patched version
        print("TESTING WITH PATCHED VERSION")
        patched_loader = get_text_classification_loader(
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            max_length=MAX_LENGTH
        )
        
        patched_dataset = patched_loader.load_huggingface_dataset(
            dataset_name=DATASET_NAME,
            text_column="text",
            label_column="label"
        )
        
        # Check first example
        print_separator()
        print("PATCHED DATASET FIRST EXAMPLE")
        first_patched = patched_dataset["train"][0]
        print(f"Type: {type(first_patched)}")
        
        if hasattr(first_patched, 'keys'):
            print(f"Keys: {first_patched.keys()}")
            if 'text' in first_patched:
                print(f"Text: {first_patched['text'][:50]}...")
            if 'label' in first_patched:
                print(f"Label: {first_patched['label']}")
        else:
            print(f"Not a dictionary-like object: {repr(first_patched)}")
            
    except Exception as e:
        print(f"Error with patched version: {type(e).__name__}: {str(e)}")
        # Restore original method
        CustomNLPDataset.__getitem__ = original_getitem
    
if __name__ == "__main__":
    debug_dataset_handling()
