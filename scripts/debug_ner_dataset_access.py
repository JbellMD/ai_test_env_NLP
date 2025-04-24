#!/usr/bin/env python
"""
Debug script to trace NER dataset structure from loading to final access.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from src.data.preprocessing import TextPreprocessor
from src.data.data_loader import get_ner_loader, NLPDatasetLoader

def print_separator():
    print("\n" + "="*80 + "\n")

def debug_ner_dataset_processing():
    # Configuration matching the notebook
    MODEL_NAME = "dslim/bert-base-NER"
    DATASET_NAME = "conll2003"
    MAX_LENGTH = 128
    
    # 1. Initialize components
    print("Step 1: Initializing components")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    preprocessor = TextPreprocessor()
    
    # 2. Create raw dataset loader without wrapping
    print("\nStep 2: Creating raw dataset loader")
    raw_loader = NLPDatasetLoader(
        preprocessor=preprocessor,
        task_type="ner"
    )
    
    # 3. Load dataset directly
    print("\nStep 3: Loading raw dataset")
    raw_dataset = raw_loader.load_huggingface_dataset(
        dataset_name=DATASET_NAME,
        text_column="tokens",
        label_column="ner_tags",
        trust_remote_code=True
    )
    
    # 4. Examine raw dataset structure
    print("\nStep 4: Examining raw dataset structure")
    print(f"Dataset type: {type(raw_dataset)}")
    print(f"Dataset keys: {raw_dataset.keys()}")
    print(f"Train split example type: {type(raw_dataset['train'][0])}")
    
    # Check what happens when we access a specific example
    example = raw_dataset['train'][0]
    print(f"Example keys: {example.keys()}")
    print(f"'tokens' key type: {type(example['tokens'])}")
    print(f"'tokens' first 10 elements: {example['tokens'][:10]}")
    
    # 5. Create dataset loader with tokenizer
    print_separator()
    print("Step 5: Creating dataset loader with tokenizer")
    dataset_loader = get_ner_loader(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_length=MAX_LENGTH
    )
    
    # 6. Load dataset with loader
    print("\nStep 6: Loading dataset with loader")
    processed_dataset = dataset_loader.load_huggingface_dataset(
        dataset_name=DATASET_NAME,
        text_column="tokens",
        label_column="ner_tags"
    )
    
    # 7. Examine processed dataset structure
    print("\nStep 7: Examining processed dataset structure")
    print(f"Dataset type: {type(processed_dataset)}")
    print(f"Dataset keys: {processed_dataset.keys()}")
    
    # Check what happens when we access a specific example
    processed_example = processed_dataset['train'][0]
    print(f"Processed example type: {type(processed_example)}")
    
    # Try to access tokens directly
    print("\nTrying to access 'tokens' field:")
    try:
        tokens = processed_example['tokens']
        print(f"Tokens: {tokens[:10]}...")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
    
    # Try with different methods
    print("\nTrying alternative access methods:")
    if hasattr(processed_example, 'keys'):
        print(f"Example keys: {processed_example.keys()}")
    else:
        print("Example doesn't have keys() method")
        
    if isinstance(processed_example, str):
        print(f"Example is a string of length: {len(processed_example)}")
        print(f"First 50 chars: {processed_example[:50]}...")
    
    # 8. Check the prepare_dataset method 
    print_separator()
    print("Step 8: Checking prepare_dataset method")
    
    # Check what happens in prepare_dataset
    print("\nPreparing dataset for training:")
    try:
        prepared = dataset_loader.prepare_dataset(
            processed_dataset, 
            split="train", 
            batch_size=16, 
            shuffle=True
        )
        print(f"Prepared dataset type: {type(prepared)}")
        
        # Try to get a batch
        batch = next(iter(prepared))
        print(f"Batch keys: {batch.keys()}")
        
    except Exception as e:
        print(f"Error in prepare_dataset: {type(e).__name__}: {str(e)}")
    
if __name__ == "__main__":
    debug_ner_dataset_processing()
