"""
Data loading utilities for NLP tasks.

This module provides classes and functions to load and preprocess
data from various sources and formats for NLP tasks.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset

from .preprocessing import TextPreprocessor, TokenizerWrapper


class NLPDatasetLoader:
    """Class for loading and preparing datasets for NLP tasks."""

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        tokenizer_wrapper: Optional[TokenizerWrapper] = None,
        task_type: str = "classification",
    ):
        """
        Initialize the dataset loader.

        Args:
            preprocessor: Text preprocessor instance
            tokenizer_wrapper: TokenizerWrapper instance
            task_type: Type of NLP task (classification, ner, summarization, etc.)
        """
        self.preprocessor = preprocessor
        self.tokenizer_wrapper = tokenizer_wrapper
        self.task_type = task_type
        self.label_map = None
        self.text_column = None
        self.label_column = None

    def load_huggingface_dataset(
        self,
        dataset_name: str,
        text_column: str = "text",
        label_column: str = "label",
        split: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from Hugging Face's datasets.

        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            text_column: Name of the column containing the text
            label_column: Name of the column containing labels
            split: Dataset split to load (train, validation, test)
            trust_remote_code: Whether to trust remote code when loading datasets (needed for some datasets)

        Returns:
            Loaded dataset or dataset dictionary
        """
        # Load the dataset
        try:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code)
            
            # Store the columns for later reference
            if text_column != "text" or label_column != "label":
                self.text_column = text_column
                self.label_column = label_column
            
            # For the IMDB dataset (and similar ones), we might not need further processing
            # since the columns are already named 'text' and 'label'
            
            # Create label map for classification tasks (only if we have labels)
            if self.task_type == "classification":
                if isinstance(dataset, DatasetDict) and "train" in dataset:
                    # Get unique labels from training set
                    if label_column in dataset["train"].column_names:
                        labels = sorted(set(dataset["train"][label_column]))
                        self.label_map = {i: label for i, label in enumerate(labels)}
                elif isinstance(dataset, Dataset) and label_column in dataset.column_names:
                    # Get unique labels from the dataset
                    labels = sorted(set(dataset[label_column]))
                    self.label_map = {i: label for i, label in enumerate(labels)}
                    
            # Return the dataset without further modification
            return dataset
            
        except Exception as e:
            raise ValueError(f"Error loading dataset {dataset_name}: {str(e)}")
        
    def _process_dataset_split(self, dataset_split, text_column, label_column):
        """
        Process a single dataset split to ensure it has the required format.
        
        Args:
            dataset_split: Dataset split to process
            text_column: Name of the column containing the text
            label_column: Name of the column containing labels
            
        Returns:
            Processed dataset split
        """
        # Verify columns exist
        column_names = dataset_split.column_names
        
        # For NER and token classification tasks, we may have special column naming
        if self.task_type == "ner":
            # NER datasets typically have 'tokens' and 'ner_tags' columns
            # We'll preserve the original structure for these
            return dataset_split
        
        # Check if columns exist (for other task types)
        has_text_column = text_column in column_names
        has_label_column = label_column in column_names
        
        if not has_text_column:
            raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {column_names}")
        
        # Only rename columns if necessary, and use the map function in a way that preserves structure
        if text_column != "text" or (label_column != "label" and has_label_column):
            # Define a function that preserves all fields and just adds/renames necessary ones
            def map_columns(example):
                # Make a copy to avoid modifying the original
                result = dict(example)
                # Add the standardized field names
                result["text"] = example[text_column]
                if has_label_column:
                    result["label"] = example[label_column]
                return result
            
            # Apply the mapping with batched=False to ensure we process one example at a time
            dataset_split = dataset_split.map(map_columns, batched=False)
            
        # The dataset is now properly structured
        return dataset_split

    def load_from_csv(
        self,
        file_path: str,
        text_column: str = "text",
        label_column: Optional[str] = "label",
        delimiter: str = ",",
    ) -> Dataset:
        """
        Load dataset from a CSV file.

        Args:
            file_path: Path to the CSV file
            text_column: Name of the column containing the text
            label_column: Name of the column containing labels
            delimiter: CSV delimiter

        Returns:
            Loaded dataset
        """
        df = pd.read_csv(file_path, delimiter=delimiter)

        # For classification, map labels to integers if they aren't already
        if self.task_type == "classification" and label_column is not None:
            if not pd.api.types.is_numeric_dtype(df[label_column]):
                labels = sorted(df[label_column].unique())
                self.label_map = {i: label for i, label in enumerate(labels)}
                label_to_id = {label: i for i, label in enumerate(labels)}
                df["label"] = df[label_column].map(label_to_id)
            else:
                df["label"] = df[label_column]

        # Create 'text' column if it doesn't match
        if text_column != "text":
            df["text"] = df[text_column]

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        return dataset

    def load_from_json(
        self, file_path: str, text_key: str = "text", label_key: Optional[str] = "label"
    ) -> Dataset:
        """
        Load dataset from a JSON file.

        Args:
            file_path: Path to the JSON file
            text_key: Key for the text field
            label_key: Key for the label field

        Returns:
            Loaded dataset
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to DataFrame then to Dataset
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            # If JSON is not a list of records, try to normalize it
            df = pd.json_normalize(data)

        # Continue as in load_from_csv
        if text_key != "text":
            df["text"] = df[text_key]

        if self.task_type == "classification" and label_key is not None:
            if not pd.api.types.is_numeric_dtype(df[label_key]):
                labels = sorted(df[label_key].unique())
                self.label_map = {i: label for i, label in enumerate(labels)}
                label_to_id = {label: i for i, label in enumerate(labels)}
                df["label"] = df[label_key].map(label_to_id)
            else:
                df["label"] = df[label_key]

        dataset = Dataset.from_pandas(df)
        return dataset

    def preprocess_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        apply_preprocessing: bool = True,
        tokenize: bool = True,
        preprocessing_fn: Optional[Callable] = None,
    ) -> Union[Dataset, DatasetDict]:
        """
        Preprocess a dataset by applying text preprocessing and tokenization.
        
        Args:
            dataset: Dataset to preprocess
            apply_preprocessing: Whether to apply text preprocessing
            tokenize: Whether to apply tokenization
            preprocessing_fn: Optional custom preprocessing function to use
            
        Returns:
            Preprocessed dataset
        """
        # For NER datasets, we don't need to preprocess the text
        if self.task_type == "ner":
            # Skip text preprocessing for NER - token-based datasets
            # But we can still apply tokenization if needed
            if tokenize and self.tokenizer_wrapper is not None:
                def tokenize_tokens(examples):
                    # Get tokens from the dataset
                    tokens = examples["tokens"]
                    
                    # Tokenize the tokens
                    encoding = self.tokenizer_wrapper.tokenize(
                        tokens,
                        is_split_into_words=True,
                        return_tensors="pt"
                    )
                    
                    # Add labels if they exist
                    if "ner_tags" in examples:
                        encoding["labels"] = examples["ner_tags"]
                        
                    return encoding
                
                dataset = dataset.map(tokenize_tokens, batched=True)
            
            return dataset
        
        # For other task types (classification, summarization, etc.)
        # Apply text preprocessing if needed
        if apply_preprocessing and self.preprocessor is not None:
            # Either use custom preprocessing function or default batch preprocessing
            if preprocessing_fn is not None:
                def preprocess_text(examples):
                    examples["text"] = preprocessing_fn(examples["text"])
                    return examples
            else:
                def preprocess_text(examples):
                    examples["text"] = self.preprocessor.batch_preprocess(examples["text"])
                    return examples
                
            dataset = dataset.map(preprocess_text, batched=True)
        
        # Apply tokenization if needed
        if tokenize and self.tokenizer_wrapper is not None:
            dataset = self.tokenizer_wrapper.tokenize_dataset(dataset)
        
        return dataset

    def create_torch_dataloaders(
        self,
        dataset: Union[Dataset, DatasetDict],
        batch_size: int = 16,
        shuffle_train: bool = True,
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoader objects for each split in the dataset.

        Args:
            dataset: Input dataset
            batch_size: Batch size for the DataLoader
            shuffle_train: Whether to shuffle the training data

        Returns:
            Dictionary of DataLoader objects for each split
        """
        dataloaders = {}

        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                shuffle = shuffle_train and split_name == "train"
                if self.task_type == "ner":
                    dataloaders[split_name] = DataLoader(
                        split_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=token_classification_collate_fn
                    )
                else:
                    dataloaders[split_name] = DataLoader(
                        split_dataset, batch_size=batch_size, shuffle=shuffle
                    )
        else:
            # Single dataset, assume it's for training
            if self.task_type == "ner":
                dataloaders["train"] = DataLoader(
                    dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=token_classification_collate_fn
                )
            else:
                dataloaders["train"] = DataLoader(
                    dataset, batch_size=batch_size, shuffle=shuffle_train
                )

        return dataloaders


def token_classification_collate_fn(batch):
    """
    Custom collate function for token classification (NER) batches.
    
    Args:
        batch: List of examples from dataset
    
    Returns:
        Batched inputs with properly padded tensors
    """
    # Safely skip empty batches
    if not batch or not isinstance(batch, list):
        return {}
    
    # Skip this batch if it doesn't contain all required fields
    required_keys = {'input_ids', 'attention_mask'}
    if not all(all(k in example for k in required_keys) for example in batch):
        # Return empty dict if batch doesn't contain required keys
        for example in batch:
            if set(example.keys()) != set(batch[0].keys()):
                print(f"Warning: Inconsistent keys in batch. Example keys: {example.keys()}")
        return {}
    
    # Collect all keys from batch
    all_keys = set(k for example in batch for k in example.keys())
    
    # Create a new batch dict
    batch_dict = {}
    
    # Process each key separately
    for key in all_keys:
        # Skip if any example doesn't have this key
        if not all(key in example for example in batch):
            continue
        
        # Get all values for this key
        values = [example[key] for example in batch]
        
        # Skip keys with string values or other non-numeric types
        if any(isinstance(v, str) for v in values):
            continue
            
        # Convert to tensors if needed
        try:
            values = [
                torch.tensor(v, dtype=torch.long) if not isinstance(v, torch.Tensor) else v
                for v in values
            ]
        except (ValueError, TypeError) as e:
            # Skip this key if we can't convert to tensors
            print(f"Warning: Could not convert values for key '{key}' to tensors: {e}")
            continue
        
        # Handle different sized tensors
        if key in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']:
            # Find max length in batch for this key
            max_length = max(len(v) for v in values)
            
            # Pad values
            padded_values = []
            for v in values:
                pad_size = max_length - len(v)
                
                if pad_size > 0:
                    pad_val = 0  # Default pad value 
                    if key == 'labels':
                        pad_val = -100  # Ignore index for loss calculation
                        
                    # Create padding tensor with appropriate size and value
                    padding = torch.full((pad_size,), pad_val, dtype=v.dtype)
                    padded_v = torch.cat([v, padding])
                else:
                    padded_v = v
                
                padded_values.append(padded_v)
            
            # Stack padded values
            try:
                batch_dict[key] = torch.stack(padded_values)
            except RuntimeError as e:
                # If stacking fails, print debug info
                sizes = [v.size() for v in padded_values]
                print(f"Warning: Failed to stack {key} values with shapes: {sizes}")
                if key == 'labels':
                    # For labels, this is optional, so we can skip
                    continue
                else:
                    # Re-raise if it's a required field
                    raise
        else:
            # For other keys, only attempt stacking if all tensors have same shape
            try:
                shapes = set(v.shape for v in values)
                if len(shapes) == 1:
                    batch_dict[key] = torch.stack(values)
            except (RuntimeError, ValueError, AttributeError):
                # Skip keys with inconsistent shapes or non-tensor values
                continue
    
    return batch_dict


class CustomNLPDataset(TorchDataset):
    """Custom PyTorch Dataset for NLP tasks."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of text inputs
            labels: List of labels (optional)
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            preprocessor: Text preprocessor instance
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor

        # Preprocess texts if needed
        if self.preprocessor is not None:
            self.texts = self.preprocessor.batch_preprocess(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing text and optional labels
        """
        # Get the text
        text = self.texts[idx]
        
        # Create the return dictionary
        item = {"text": text}
        
        # Add tokenization if tokenizer is provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # Add encoding fields to the item dictionary
            for k, v in encoding.items():
                item[k] = v.squeeze(0)
        
        # Add label if available
        if self.labels is not None:
            item["label"] = self.labels[idx]
            
        return item


# Factory functions for different task types
def get_text_classification_loader(tokenizer, preprocessor=None, max_length=512, **kwargs):
    """
    Get a dataset loader configured for text classification.
    
    Args:
        tokenizer: Tokenizer to use
        preprocessor: Optional text preprocessor
        max_length: Maximum sequence length
        **kwargs: Additional arguments to pass to TokenizerWrapper
    """
    # Update kwargs with length parameters
    tokenizer_kwargs = {
        'max_length': max_length,
        **kwargs
    }
    
    return NLPDatasetLoader(
        preprocessor=preprocessor,
        tokenizer_wrapper=TokenizerWrapper(tokenizer, **tokenizer_kwargs),
        task_type="classification",
    )


def get_ner_loader(tokenizer, preprocessor=None, max_length=512, **kwargs):
    """
    Get a dataset loader configured for named entity recognition.
    
    Args:
        tokenizer: Tokenizer to use
        preprocessor: Optional text preprocessor
        max_length: Maximum sequence length
        **kwargs: Additional arguments to pass to TokenizerWrapper
    """
    # Update kwargs with length parameters
    tokenizer_kwargs = {
        'max_length': max_length,
        **kwargs
    }
    
    return NLPDatasetLoader(
        preprocessor=preprocessor,
        tokenizer_wrapper=TokenizerWrapper(tokenizer, **tokenizer_kwargs),
        task_type="ner",
    )


def get_summarization_loader(tokenizer, preprocessor=None, max_input_length=512, max_output_length=128, **kwargs):
    """
    Get a dataset loader configured for text summarization.
    
    Args:
        tokenizer: Tokenizer to use
        preprocessor: Optional text preprocessor
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        **kwargs: Additional arguments to pass to TokenizerWrapper
    """
    # Update kwargs with length parameters
    tokenizer_kwargs = {
        'max_length': max_input_length,
        **kwargs
    }
    
    return NLPDatasetLoader(
        preprocessor=preprocessor,
        tokenizer_wrapper=TokenizerWrapper(tokenizer, **tokenizer_kwargs),
        task_type="summarization",
    )
