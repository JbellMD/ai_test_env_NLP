"""
Model registry for NLP models.

This module manages the loading, caching, and access to various NLP models
used by the API.
"""
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer

from ..utils.logging_utils import get_logger
from ..models.classifier import TransformerClassifier, create_classifier
from ..models.named_entity_recognition import NERModel, create_ner_model
from ..models.sentiment_analyzer import SentimentAnalyzer, create_sentiment_analyzer, AspectBasedSentimentAnalyzer
from ..models.summarizer import TextSummarizer, ExtractiveSummarizer, ControlledSummarizer

# Initialize logger
logger = get_logger(__name__)

# Model registry
_MODEL_REGISTRY = {}
_TOKENIZER_REGISTRY = {}

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
        "metrics": {"accuracy": 0.91, "f1": 0.90}
    },
    "bert_classification": {
        "model_name": "bert-base-uncased",
        "task": "classification",
        "num_labels": 3,
        "description": "BERT model for general text classification",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.89, "f1": 0.88}
    },
    "roberta_classification": {
        "model_name": "roberta-base",
        "task": "classification",
        "num_labels": 3,
        "description": "RoBERTa model for general text classification",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.92, "f1": 0.91}
    },
    
    # NER models
    "default_ner": {
        "model_name": "dslim/bert-base-NER",
        "task": "ner",
        "num_labels": 9,
        "description": "BERT-based model fine-tuned for Named Entity Recognition",
        "size": "base",
        "languages": ["en"],
        "metrics": {"f1": 0.89, "precision": 0.88, "recall": 0.90}
    },
    "spacy_ner": {
        "model_name": "en_core_web_sm",
        "task": "ner",
        "description": "SpaCy English NER model",
        "size": "small",
        "languages": ["en"],
        "metrics": {"f1": 0.86, "precision": 0.85, "recall": 0.87},
        "is_spacy": True
    },
    
    # Sentiment analysis models
    "default_sentiment": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "sentiment",
        "num_labels": 2,
        "description": "DistilBERT model fine-tuned on SST-2 for sentiment analysis",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.91, "f1": 0.90}
    },
    "twitter_sentiment": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment",
        "task": "sentiment",
        "num_labels": 3,
        "description": "RoBERTa model fine-tuned on Twitter data for sentiment analysis",
        "size": "base",
        "languages": ["en"],
        "metrics": {"accuracy": 0.92, "f1": 0.91}
    },
    
    # Summarization models
    "default_summarization": {
        "model_name": "facebook/bart-large-cnn",
        "task": "summarization",
        "description": "BART model fine-tuned on CNN/DM for abstractive summarization",
        "size": "large",
        "languages": ["en"],
        "metrics": {"rouge1": 0.44, "rouge2": 0.21, "rougeL": 0.41}
    },
    "t5_summarization": {
        "model_name": "t5-small",
        "task": "summarization",
        "description": "T5 model for abstractive summarization",
        "size": "small",
        "languages": ["en"],
        "metrics": {"rouge1": 0.41, "rouge2": 0.19, "rougeL": 0.38}
    },
    "extractive_tfidf": {
        "model_name": "tfidf",
        "task": "summarization",
        "description": "TF-IDF based extractive summarization",
        "size": "tiny",
        "languages": ["en", "multilingual"],
        "metrics": {"rouge1": 0.38, "rouge2": 0.15, "rougeL": 0.35},
        "is_extractive": True
    },
    "extractive_textrank": {
        "model_name": "textrank",
        "task": "summarization",
        "description": "TextRank based extractive summarization",
        "size": "tiny",
        "languages": ["en", "multilingual"],
        "metrics": {"rouge1": 0.37, "rouge2": 0.14, "rougeL": 0.34},
        "is_extractive": True
    }
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
                model_name=config["model_name"],
                num_labels=config.get("num_labels", 2)
            )
        
        elif config["task"] == "ner":
            if config.get("is_spacy", False):
                # Special case for SpaCy models
                import spacy
                model = spacy.load(config["model_name"])
            else:
                model = create_ner_model(
                    model_name=config["model_name"],
                    num_labels=config.get("num_labels", 9)
                )
        
        elif config["task"] == "sentiment":
            model = create_sentiment_analyzer(
                model_name=config["model_name"],
                num_labels=config.get("num_labels", 3)
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
            models.append(ModelInfo(
                name=name,
                task=config["task"],
                description=config.get("description"),
                size=config.get("size"),
                languages=config.get("languages"),
                metrics=config.get("metrics")
            ))
    
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
