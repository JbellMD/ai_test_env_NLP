"""
Data preprocessing utilities for NLP tasks.

This module provides functions for text preprocessing, tokenization,
and data cleaning specific to different NLP tasks.
"""

import re
import unicodedata
from typing import Callable, Dict, List, Optional, Union

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datasets import Dataset, DatasetDict

# Download necessary NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class TextPreprocessor:
    """Class for text preprocessing operations."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        remove_whitespace: bool = True,
        lemmatize: bool = False,
        language: str = "english",
    ):
        """
        Initialize the text preprocessor with specific options.

        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            remove_stopwords: Whether to remove stopwords
            remove_whitespace: Whether to normalize whitespace
            lemmatize: Whether to apply lemmatization
            language: Language for stopwords and lemmatization
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords_flag = remove_stopwords  # Rename to avoid conflict with method
        self.remove_whitespace = remove_whitespace
        self.lemmatize = lemmatize
        self.language = language

        # Initialize stopwords if needed
        if self.remove_stopwords_flag:
            self.stopwords_set = set(stopwords.words(language))

        # Initialize lemmatizer if needed
        if self.lemmatize:
            from nltk.stem import WordNetLemmatizer

            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet")
            self.lemmatizer = WordNetLemmatizer()

    def normalize_text(self, text: str) -> str:
        """
        Normalize text (lowercase).
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        return text.lower() if self.lowercase else text
    
    def remove_special_chars(self, text: str) -> str:
        """
        Remove special characters from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
        """
        # Remove special characters but keep regular punctuation
        return re.sub(r'[@#$%^&*()_+=[\]{}|\\<>~`]', '', text)
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        tokens = word_tokenize(text)
        if not hasattr(self, 'stopwords_set'):
            self.stopwords_set = set(stopwords.words(self.language))
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords_set]
        return ' '.join(filtered_tokens)
    
    def preprocess_text(self, text: str, normalize: bool = True, 
                      remove_special_chars: bool = False, 
                      remove_stopwords: bool = False) -> str:
        """
        Preprocess text with custom options.
        
        Args:
            text: Input text to preprocess
            normalize: Whether to normalize the text
            remove_special_chars: Whether to remove special characters
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Preprocessed text
        """
        # Apply preprocessing steps based on arguments
        if normalize:
            text = self.normalize_text(text)
        
        if remove_special_chars:
            text = self.remove_special_chars(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        return text

    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to the input text.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # Tokenize
        tokens = word_tokenize(text)

        if self.remove_stopwords_flag:  # Use the renamed flag
            tokens = [token for token in tokens if token not in self.stopwords_set]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Rejoin tokens
        text = " ".join(tokens)

        if self.remove_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Apply preprocessing to a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class TokenizerWrapper:
    """Wrapper for different tokenizers to provide a unified interface."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ):
        """
        Initialize the tokenizer wrapper.

        Args:
            tokenizer: The base tokenizer to wrap (e.g., from Hugging Face)
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length', 'longest', etc.)
            truncation: Whether to truncate sequences longer than max_length
            return_tensors: Type of tensors to return ('pt', 'tf', 'np')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors

    def encode(self, texts: Union[str, List[str]], **kwargs) -> Dict:
        """
        Encode texts with the tokenizer.

        Args:
            texts: Input text or list of texts
            **kwargs: Additional arguments to pass to the tokenizer

        Returns:
            Dictionary of tokenized outputs
        """
        # Override default kwargs with any that were passed
        tokenizer_kwargs = {
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "return_tensors": self.return_tensors,
        }
        tokenizer_kwargs.update(kwargs)

        # Tokenize the texts
        return self.tokenizer(texts, **tokenizer_kwargs)
        
    def tokenize(self, texts: Union[str, List[str]], is_split_into_words: bool = False, **kwargs) -> Dict:
        """
        Tokenize texts with support for word-split inputs (for NER).

        Args:
            texts: Input text or list of texts
            is_split_into_words: If True, texts is already split into words (for NER)
            **kwargs: Additional arguments to pass to the tokenizer

        Returns:
            Dictionary of tokenized outputs
        """
        # Override default kwargs with any that were passed
        tokenizer_kwargs = {
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "return_tensors": self.return_tensors,
            "is_split_into_words": is_split_into_words
        }
        tokenizer_kwargs.update(kwargs)

        # Tokenize the texts
        return self.tokenizer(texts, **tokenizer_kwargs)

    def tokenize_dataset(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        """
        Tokenize an entire dataset.

        Args:
            dataset: Dataset to tokenize

        Returns:
            Tokenized dataset
        """
        def tokenize_text(examples):
            # For token classification (NER), check if 'tokens' is available
            if 'tokens' in examples:
                tokenized = self.tokenize(
                    examples['tokens'], 
                    is_split_into_words=True
                )
                # Also add the labels if they exist
                if 'ner_tags' in examples:
                    tokenized['labels'] = examples['ner_tags']
                return tokenized
            # For other tasks, use the standard text column
            elif 'text' in examples:
                return self.encode(examples['text'])
            else:
                # Try to handle other structures
                columns = list(examples.keys())
                raise ValueError(f"Cannot find 'text' or 'tokens' column in dataset. Found: {columns}")

        # Apply tokenization to the dataset
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: dataset[split].map(tokenize_text, batched=True)
                for split in dataset
            })
        else:
            return dataset.map(tokenize_text, batched=True)

    def decode(self, token_ids, **kwargs):
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, **kwargs)

    def __getattr__(self, name):
        """Forward any other attributes to the wrapped tokenizer."""
        return getattr(self.tokenizer, name)


# Utility functions for data augmentation
def synonym_replacement(text: str, n: int = 1) -> str:
    """Replace n words in the text with their synonyms."""
    # Implementation would use WordNet or similar resource
    return text  # Placeholder


def random_insertion(text: str, n: int = 1) -> str:
    """Insert n random words into the text."""
    # Implementation would insert words from vocabulary
    return text  # Placeholder


def random_swap(text: str, n: int = 1) -> str:
    """Randomly swap positions of n pairs of words in the text."""
    words = text.split()
    if len(words) <= 1:
        return text

    # Rest of implementation
    return text  # Placeholder


def random_deletion(text: str, p: float = 0.1) -> str:
    """Randomly delete words from the text with probability p."""
    words = text.split()
    if len(words) <= 1:
        return text

    # Rest of implementation
    return text  # Placeholder
