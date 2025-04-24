"""
Model registry for NLP models.

This module manages the loading, caching, and access to various NLP models
used by the API.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from ..models.classifier import TransformerClassifier, create_classifier
from ..models.named_entity_recognition import NERModel, create_ner_model
from ..models.sentiment_analyzer import (
    AspectBasedSentimentAnalyzer,
    SentimentAnalyzer,
    create_sentiment_analyzer,
)
from ..models.summarizer import (
    ControlledSummarizer,
    ExtractiveSummarizer,
    TextSummarizer,
)
from ..utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Model registry
_MODEL_REGISTRY: Dict[str, Any] = {}
_TOKENIZER_REGISTRY: Dict[str, Any] = {}

class ModelRegistry:
    """
    Registry for storing and retrieving NLP models.
    
    This class provides a façade over the module-level functions for
    tests and external code that expect a class-based interface.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the model registry.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self._cached_models = {}
        
        # Initialize from model directory if provided
        if model_dir and os.path.isdir(model_dir):
            self._scan_model_directory()
    
    def _scan_model_directory(self):
        """Scan the model directory for available models and their metadata."""
        if not self.model_dir:
            return
        
        self._cached_models = {}
        
        # Look for model subdirectories
        for item in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, item)
            
            if os.path.isdir(model_path):
                # Check for metadata file
                metadata_path = os.path.join(model_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            metadata["id"] = item  # Add model ID from directory name
                            self._cached_models[item] = metadata
                    except Exception as e:
                        logger.warning(f"Error loading metadata for {item}: {e}")
    
    def get_available_models(self):
        """Get a list of all available models."""
        if self.model_dir:
            return list(self._cached_models.values())
        else:
            # Fall back to global registry if no model_dir
            return list_available_models()
    
    def get_models_by_task(self, task):
        """
        Get models filtered by task.
        
        Args:
            task: Task type to filter by
            
        Returns:
            List of models for the specified task
        """
        if self.model_dir:
            return [
                model for model in self._cached_models.values() 
                if model.get("task") == task
            ]
        else:
            # Fall back to global registry
            models = list_available_models(task)
            return [model for model in models if model.task == task]
    
    def get_model_metadata(self, model_id):
        """
        Get metadata for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata dictionary
            
        Raises:
            ValueError: If model is not found
        """
        if self.model_dir:
            if model_id in self._cached_models:
                return self._cached_models[model_id]
            else:
                raise ValueError(f"Model not found: {model_id}")
        else:
            # Fall back to global registry
            for model in list_available_models():
                if model.name == model_id:
                    return {
                        "id": model_id,
                        "task": model.task,
                        "name": model.name,
                        "description": model.description,
                        "metrics": model.metrics
                    }
            raise ValueError(f"Model not found: {model_id}")
    
    def register_model(self, model_path, metadata):
        """
        Register a new model.
        
        Args:
            model_path: Path to the model directory
            metadata: Model metadata dictionary
            
        Returns:
            Model ID
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        
        # Get model ID from directory name
        model_id = os.path.basename(model_path)
        
        # Save metadata
        metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update cache
        metadata["id"] = model_id
        self._cached_models[model_id] = metadata
        
        return model_id
    
    def get_model(self, model_name: str, task: Optional[str] = None):
        """Get a model by name and task."""
        return get_model(model_name, task)
    
    def get_tokenizer(self, model_name: str, task: Optional[str] = None):
        """Get a tokenizer by model name and task."""
        return get_tokenizer(model_name, task)
    
    def list_available_models(self, task: Optional[str] = None):
        """List available models, optionally filtered by task."""
        return list_available_models(task)
    
    def initialize_models(self):
        """Initialize default models on startup."""
        return initialize_models()
    
    def cleanup_models(self):
        """Clean up models on shutdown."""
        return cleanup_models()

# Model configurations
MODEL_CONFIGS = {
    # Classification models
    "default_classification": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "classification",
        "num_labels": 2,
        "description": "DistilBERT model fine-tuned on SST-2 for sentiment classification",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.91, "f1": 0.90},
    },
    "bert_classification": {
        "model_name": "bert-base-uncased",
        "task": "classification",
        "num_labels": 3,
        "description": "BERT model for general text classification",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.89, "f1": 0.88},
    },
    "roberta_classification": {
        "model_name": "roberta-base",
        "task": "classification",
        "num_labels": 3,
        "description": "RoBERTa model for general text classification",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.92, "f1": 0.91},
    },
    # NER models
    "default_ner": {
        "model_name": "dslim/bert-base-NER",
        "task": "ner",
        "num_labels": 9,
        "description": "BERT-based model fine-tuned for Named Entity Recognition",
        "size": "base",
        "languages": ["en"],
        "metrics": {"f1": 0.89, "precision": 0.88, "recall": 0.90},
    },
    "spacy_ner": {
        "model_name": "en_core_web_sm",
        "task": "ner",
        "description": "SpaCy English NER model",
        "size": "small",
        "languages": ["en"],
        "metrics": {"f1": 0.86, "precision": 0.85, "recall": 0.87},
        "is_spacy": True,
    },
    # Sentiment analysis models
    "default_sentiment": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "sentiment",
        "num_labels": 2,
        "description": "DistilBERT model fine-tuned on SST-2 for sentiment analysis",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.91, "f1": 0.90},
    },
    "twitter_sentiment": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment",
        "task": "sentiment",
        "num_labels": 3,
        "description": "RoBERTa model fine-tuned on Twitter data for sentiment analysis",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.92, "f1": 0.91},
    },
    # Summarization models
    "default_summarization": {
        "model_name": "facebook/bart-large-cnn",
        "task": "summarization",
        "description": "BART model fine-tuned on CNN/DM for abstractive summarization",
        "size": "large",
        "languages": ["en"],
        "metrics": {"rouge1": 0.44, "rouge2": 0.21, "rougeL": 0.41},
    },
    "t5_summarization": {
        "model_name": "t5-small",
        "task": "summarization",
        "description": "T5 model for abstractive summarization",
        "size": "small",
        "languages": ["en"],
        "metrics": {"rouge1": 0.41, "rouge2": 0.19, "rougeL": 0.38},
    },
    "extractive_tfidf": {
        "model_name": "tfidf",
        "task": "summarization",
        "description": "TF-IDF based extractive summarization",
        "size": "tiny",
        "languages": ["en", "multilingual"],
        "metrics": {"rouge1": 0.38, "rouge2": 0.15, "rougeL": 0.35},
        "is_extractive": True,
    },
    "extractive_textrank": {
        "model_name": "textrank",
        "task": "summarization",
        "description": "TextRank based extractive summarization",
        "size": "tiny",
        "languages": ["en", "multilingual"],
        "metrics": {"rouge1": 0.37, "rouge2": 0.14, "rougeL": 0.34},
        "is_extractive": True,
    },
}


class ModelInfo(BaseModel):
    """Model for information about an available model."""

    name: str
    task: str
    description: Optional[str] = None
    size: Optional[str] = None
    languages: Optional[List[str]] = None
    metrics: Optional[Dict[str, float]] = None


def initialize_models():
    """Initialize default models on startup."""
    logger.info("Initializing default models")

    # Force loading of default models
    get_model("default_classification", "classification")
    get_model("default_ner", "ner")
    get_model("default_sentiment", "sentiment")

    # Don't preload summarization models as they're usually larger
    logger.info(f"Initialized {len(_MODEL_REGISTRY)} default models")


def cleanup_models():
    """Clean up models on shutdown."""
    logger.info("Cleaning up models")

    # Clear registries
    _MODEL_REGISTRY.clear()
    _TOKENIZER_REGISTRY.clear()

    # Force garbage collection
    import gc

    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Model cleanup complete")


def get_model(model_name: str, task: Optional[str] = None):
    """
    Get a model by name and task.

    Args:
        model_name: Name of the model
        task: Task type (classification, ner, sentiment, summarization)

    Returns:
        Model instance or None if not found
    """
    # Check if model is already loaded
    key = f"{model_name}_{task}" if task else model_name

    if key in _MODEL_REGISTRY:
        logger.debug(f"Using cached model: {key}")
        return _MODEL_REGISTRY[key]

    # Check if model config exists
    if model_name not in MODEL_CONFIGS:
        logger.warning(f"Model config not found: {model_name}")
        return None

    config = MODEL_CONFIGS[model_name]

    # Verify task if provided
    if task and config["task"] != task:
        logger.warning(f"Model {model_name} is for task {config['task']}, not {task}")
        return None

    # Load model based on task
    try:
        logger.info(f"Loading model: {model_name} for task {config['task']}")

        if config["task"] == "classification":
            model = create_classifier(
                model_name=config["model_name"], num_labels=config.get("num_labels", 2)
            )

        elif config["task"] == "ner":
            if config.get("is_spacy", False):
                # Special case for SpaCy models
                import spacy

                model = spacy.load(config["model_name"])
            else:
                model = create_ner_model(
                    model_name=config["model_name"],
                    num_labels=config.get("num_labels", 9),
                )

        elif config["task"] == "sentiment":
            model = create_sentiment_analyzer(
                model_name=config["model_name"], num_labels=config.get("num_labels", 3)
            )

        elif config["task"] == "summarization":
            if config.get("is_extractive", False):
                model = ExtractiveSummarizer(method=config["model_name"])
            else:
                model = TextSummarizer(model_name_or_path=config["model_name"])

        else:
            logger.warning(f"Unknown task: {config['task']}")
            return None

        # Cache model
        _MODEL_REGISTRY[key] = model

        return model

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
        return None


def get_tokenizer(model_name: str, task: Optional[str] = None):
    """
    Get a tokenizer by model name and task.

    Args:
        model_name: Name of the model
        task: Task type (classification, ner, sentiment, summarization)

    Returns:
        Tokenizer instance or None if not found
    """
    # Check if tokenizer is already loaded
    key = f"{model_name}_{task}" if task else model_name

    if key in _TOKENIZER_REGISTRY:
        logger.debug(f"Using cached tokenizer: {key}")
        return _TOKENIZER_REGISTRY[key]

    # Check if model config exists
    if model_name not in MODEL_CONFIGS:
        logger.warning(f"Model config not found: {model_name}")
        return None

    config = MODEL_CONFIGS[model_name]

    # Some models don't need tokenizers
    if config.get("is_extractive", False) or config.get("is_spacy", False):
        logger.debug(f"Model {model_name} doesn't use a Hugging Face tokenizer")
        return None

    # Load tokenizer
    try:
        logger.info(f"Loading tokenizer: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        # Cache tokenizer
        _TOKENIZER_REGISTRY[key] = tokenizer

        return tokenizer

    except Exception as e:
        logger.error(f"Error loading tokenizer {model_name}: {e}", exc_info=True)
        return None


def list_available_models(task: Optional[str] = None) -> List[ModelInfo]:
    """
    List available models, optionally filtered by task.

    Args:
        task: Task type to filter by

    Returns:
        List of model info objects
    """
    models = []

    for name, config in MODEL_CONFIGS.items():
        if task is None or config["task"] == task:
            models.append(
                ModelInfo(
                    name=name,
                    task=config["task"],
                    description=config.get("description"),
                    size=config.get("size"),
                    languages=config.get("languages"),
                    metrics=config.get("metrics"),
                )
            )

    return models


def register_model(name: str, model, tokenizer=None, config: Dict[str, Any] = None):
    """
    Register a custom model.

    Args:
        name: Model name
        model: Model instance
        tokenizer: Tokenizer instance
        config: Model configuration
    """
    # Add to registry
    _MODEL_REGISTRY[name] = model
    if tokenizer:
        _TOKENIZER_REGISTRY[name] = tokenizer

    # Add config if provided
    if config:
        MODEL_CONFIGS[name] = config

    logger.info(f"Registered custom model: {name}")
