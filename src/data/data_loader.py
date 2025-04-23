"""
Data loading utilities for NLP tasks.

This module provides classes and functions to load and preprocess
data from various sources and formats for NLP tasks.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader, TensorDataset, Dataset as TorchDataset
import torch

from .preprocessing import TextPreprocessor, TokenizerWrapper


class NLPDatasetLoader:
    """Class for loading and preparing datasets for NLP tasks."""
    
    def __init__(self, 
                 preprocessor: Optional[TextPreprocessor] = None,
                 tokenizer_wrapper: Optional[TokenizerWrapper] = None,
                 task_type: str = 'classification'):
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
    
    def load_huggingface_dataset(self, 
                                dataset_name: str, 
                                text_column: str = 'text',
                                label_column: str = 'label',
                                split: Optional[str] = None) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from Hugging Face's datasets.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            text_column: Name of the column containing the text
            label_column: Name of the column containing labels
            split: Dataset split to load (train, validation, test)
            
        Returns:
            Loaded dataset or dataset dictionary
        """
        dataset = load_dataset(dataset_name, split=split)
        
        # Rename columns if needed
        if text_column != 'text' or label_column != 'label':
            def rename_columns(examples):
                examples['text'] = examples[text_column]
                examples['label'] = examples[label_column]
                return examples
            
            dataset = dataset.map(rename_columns, remove_columns=[
                col for col in [text_column, label_column] 
                if col not in ['text', 'label']
            ])
        
        # Create label map for classification tasks
        if self.task_type == 'classification' and isinstance(dataset, Dataset):
            labels = sorted(set(dataset['label']))
            self.label_map = {i: label for i, label in enumerate(labels)}
            
        return dataset
    
    def load_from_csv(self, 
                     file_path: str, 
                     text_column: str = 'text',
                     label_column: Optional[str] = 'label',
                     delimiter: str = ',') -> Dataset:
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
        if self.task_type == 'classification' and label_column is not None:
            if not pd.api.types.is_numeric_dtype(df[label_column]):
                labels = sorted(df[label_column].unique())
                self.label_map = {i: label for i, label in enumerate(labels)}
                label_to_id = {label: i for i, label in enumerate(labels)}
                df['label'] = df[label_column].map(label_to_id)
            else:
                df['label'] = df[label_column]
        
        # Create 'text' column if it doesn't match
        if text_column != 'text':
            df['text'] = df[text_column]
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def load_from_json(self, 
                      file_path: str,
                      text_key: str = 'text',
                      label_key: Optional[str] = 'label') -> Dataset:
        """
        Load dataset from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            text_key: Key for the text field
            label_key: Key for the label field
            
        Returns:
            Loaded dataset
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame then to Dataset
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            # If JSON is not a list of records, try to normalize it
            df = pd.json_normalize(data)
        
        # Continue as in load_from_csv
        if text_key != 'text':
            df['text'] = df[text_key]
        
        if self.task_type == 'classification' and label_key is not None:
            if not pd.api.types.is_numeric_dtype(df[label_key]):
                labels = sorted(df[label_key].unique())
                self.label_map = {i: label for i, label in enumerate(labels)}
                label_to_id = {label: i for i, label in enumerate(labels)}
                df['label'] = df[label_key].map(label_to_id)
            else:
                df['label'] = df[label_key]
        
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def preprocess_dataset(self, 
                          dataset: Union[Dataset, DatasetDict],
                          apply_preprocessing: bool = True,
                          tokenize: bool = True) -> Union[Dataset, DatasetDict]:
        """
        Apply preprocessing and tokenization to the dataset.
        
        Args:
            dataset: Input dataset
            apply_preprocessing: Whether to apply text preprocessing
            tokenize: Whether to tokenize the texts
            
        Returns:
            Processed dataset
        """
        # Apply text preprocessing if needed
        if apply_preprocessing and self.preprocessor is not None:
            def preprocess_text(examples):
                examples['text'] = self.preprocessor.batch_preprocess(examples['text'])
                return examples
            
            dataset = dataset.map(preprocess_text, batched=True)
        
        # Apply tokenization if needed
        if tokenize and self.tokenizer_wrapper is not None:
            def tokenize_text(examples):
                return self.tokenizer_wrapper.encode(examples['text'])
            
            dataset = dataset.map(tokenize_text, batched=True)
        
        return dataset
    
    def create_torch_dataloaders(self, 
                                dataset: Union[Dataset, DatasetDict],
                                batch_size: int = 16,
                                shuffle_train: bool = True) -> Dict[str, DataLoader]:
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
                shuffle = shuffle_train and split_name == 'train'
                dataloaders[split_name] = DataLoader(
                    split_dataset, 
                    batch_size=batch_size, 
                    shuffle=shuffle
                )
        else:
            # Single dataset, assume it's for training
            dataloaders['train'] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle_train
            )
        
        return dataloaders


class CustomNLPDataset(TorchDataset):
    """Custom PyTorch Dataset for NLP tasks."""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: Optional[List[Any]] = None,
                 tokenizer: Optional[Any] = None,
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
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
        text = self.texts[idx]
        
        # Tokenize text if tokenizer is provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # Remove batch dimension
            item = {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            item = {'text': text}
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        
        return item


# Factory functions for different task types
def get_text_classification_loader(tokenizer, preprocessor=None):
    """Get a dataset loader configured for text classification."""
    return NLPDatasetLoader(
        preprocessor=preprocessor,
        tokenizer_wrapper=TokenizerWrapper(tokenizer),
        task_type='classification'
    )

def get_ner_loader(tokenizer, preprocessor=None):
    """Get a dataset loader configured for named entity recognition."""
    return NLPDatasetLoader(
        preprocessor=preprocessor,
        tokenizer_wrapper=TokenizerWrapper(tokenizer),
        task_type='ner'
    )

def get_summarization_loader(tokenizer, preprocessor=None):
    """Get a dataset loader configured for text summarization."""
    return NLPDatasetLoader(
        preprocessor=preprocessor,
        tokenizer_wrapper=TokenizerWrapper(tokenizer),
        task_type='summarization'
    )
