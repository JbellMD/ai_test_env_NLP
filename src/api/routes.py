"""
API routes for different NLP tasks.

This module defines FastAPI routers for various NLP tasks such as
classification, named entity recognition, sentiment analysis, and summarization.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from ..utils.logging_utils import get_logger
from .model_registry import get_model, get_tokenizer, list_available_models

# Initialize logger
logger = get_logger(__name__)


# Define request and response models for Classification
class ClassificationRequest(BaseModel):
    """Request model for text classification."""

    text: Union[str, List[str]] = Field(
        ..., description="Text to classify, can be a single string or list of strings"
    )
    model_name: Optional[str] = Field(
        None, description="Model name to use for classification"
    )
    return_probabilities: bool = Field(
        False, description="Whether to return class probabilities"
    )


class ClassificationResponse(BaseModel):
    """Response model for text classification."""

    predictions: List[Union[str, int]] = Field(
        ..., description="List of predicted classes"
    )
    probabilities: Optional[List[Dict[str, float]]] = Field(
        None, description="List of class probabilities"
    )
    model_name: str = Field(..., description="Model used for the prediction")


# Define request and response models for NER
class NERRequest(BaseModel):
    """Request model for named entity recognition."""

    text: Union[str, List[str]] = Field(..., description="Text for entity recognition")
    model_name: Optional[str] = Field(None, description="Model name to use for NER")
    align_to_words: bool = Field(
        True, description="Whether to align entity spans to word boundaries"
    )


class Entity(BaseModel):
    """Model for a named entity."""

    entity: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label")
    start: Optional[int] = Field(None, description="Start position in text")
    end: Optional[int] = Field(None, description="End position in text")


class NERResponse(BaseModel):
    """Response model for named entity recognition."""

    entities: List[List[Entity]] = Field(
        ..., description="List of entities for each text"
    )
    model_name: str = Field(..., description="Model used for the prediction")


# Define request and response models for Sentiment Analysis
class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""

    text: Union[str, List[str]] = Field(..., description="Text for sentiment analysis")
    model_name: Optional[str] = Field(
        None, description="Model name to use for sentiment analysis"
    )
    return_probabilities: bool = Field(
        False, description="Whether to return sentiment probabilities"
    )
    aspects: Optional[List[str]] = Field(
        None, description="Specific aspects to analyze (for aspect-based sentiment)"
    )


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""

    sentiments: List[Union[str, Dict[str, str]]] = Field(
        ..., description="List of sentiments or aspect-sentiment mappings"
    )
    probabilities: Optional[List[Dict[str, float]]] = Field(
        None, description="List of sentiment probabilities"
    )
    model_name: str = Field(..., description="Model used for the prediction")


# Define request and response models for Summarization
class SummarizationRequest(BaseModel):
    """Request model for text summarization."""

    text: Union[str, List[str]] = Field(..., description="Text to summarize")
    model_name: Optional[str] = Field(
        None, description="Model name to use for summarization"
    )
    max_length: int = Field(128, description="Maximum length of the summary")
    min_length: int = Field(30, description="Minimum length of the summary")
    method: Literal["abstractive", "extractive"] = Field(
        "abstractive", description="Summarization method"
    )
    length: Literal["short", "medium", "long"] = Field(
        "medium", description="Summary length preset"
    )
    focus: Optional[str] = Field(None, description="Topic to focus on in the summary")


class SummarizationResponse(BaseModel):
    """Response model for text summarization."""

    summaries: List[str] = Field(..., description="Generated summaries")
    model_name: str = Field(..., description="Model used for the prediction")


# Define model info model
class ModelInfo(BaseModel):
    """Model for information about an available model."""

    name: str = Field(..., description="Model name")
    task: str = Field(..., description="Task the model is trained for")
    description: Optional[str] = Field(None, description="Model description")
    size: Optional[str] = Field(None, description="Model size (e.g., 'base', 'large')")
    languages: Optional[List[str]] = Field(
        None, description="Languages supported by the model"
    )
    metrics: Optional[Dict[str, float]] = Field(
        None, description="Model performance metrics"
    )


# Create routers for each task
classification_router = APIRouter()
ner_router = APIRouter()
sentiment_router = APIRouter()
summarization_router = APIRouter()
model_hub_router = APIRouter()


# Classification routes
@classification_router.post("/predict", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify text into predefined categories."""
    try:
        # Get model and tokenizer
        model_name = request.model_name or "default_classification"
        model = get_model(model_name, "classification")
        tokenizer = get_tokenizer(model_name, "classification")

        if not model or not tokenizer:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        # Process the request
        texts = request.text if isinstance(request.text, list) else [request.text]

        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Get predictions
        predictions = model.predict(
            inputs, return_probabilities=request.return_probabilities
        )

        # Format response
        if request.return_probabilities:
            # Predictions are probabilities
            probs = predictions

            # Get class labels from model
            if hasattr(model, "id2label"):
                labels = [model.id2label[i] for i in range(len(model.id2label))]
            else:
                # Default to class indices
                labels = [str(i) for i in range(probs[0].shape[0])]

            # Format probabilities as dicts
            probabilities = []
            for prob in probs:
                prob_dict = {label: float(p) for label, p in zip(labels, prob)}
                probabilities.append(prob_dict)

            # Get most likely class
            predictions = [labels[prob.argmax()] for prob in probs]

            return ClassificationResponse(
                predictions=predictions,
                probabilities=probabilities,
                model_name=model_name,
            )
        else:
            # Predictions are class labels
            return ClassificationResponse(
                predictions=predictions, model_name=model_name, probabilities=None
            )

    except Exception as e:
        logger.error(f"Error in classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# NER routes
@ner_router.post("/predict", response_model=NERResponse)
async def recognize_entities(request: NERRequest):
    """Recognize named entities in text."""
    try:
        # Get model and tokenizer
        model_name = request.model_name or "default_ner"
        model = get_model(model_name, "ner")
        tokenizer = get_tokenizer(model_name, "ner")

        if not model or not tokenizer:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        # Process the request
        texts = request.text if isinstance(request.text, list) else [request.text]

        # Tokenize with offset mapping for entity alignment
        tokenized_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Extract offset mapping
        offset_mapping = tokenized_inputs.pop("offset_mapping", None)

        # Get predictions
        entities_list = model.predict(
            tokenized_inputs,
            tokenizer=tokenizer,
            original_texts=texts,
            align_to_words=request.align_to_words,
        )

        # Format response
        formatted_entities = []
        for text_entities in entities_list:
            # Convert to Entity objects based on prediction format
            if isinstance(text_entities, list) and all(
                isinstance(e, dict) for e in text_entities
            ):
                # Entities are already in the correct format
                entities = [Entity(**e) for e in text_entities]
            elif isinstance(text_entities, list) and all(
                isinstance(e, str) for e in text_entities
            ):
                # Only token-level labels, create simple entities
                entities = [
                    Entity(entity="", label=label, start=None, end=None)
                    for label in text_entities
                ]
            else:
                # Unexpected format
                entities = []

            formatted_entities.append(entities)

        return NERResponse(entities=formatted_entities, model_name=model_name)

    except Exception as e:
        logger.error(f"Error in NER: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Sentiment Analysis routes
@sentiment_router.post("/predict", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment in text."""
    try:
        # Get model and tokenizer
        model_name = request.model_name or "default_sentiment"
        model = get_model(model_name, "sentiment")
        tokenizer = get_tokenizer(model_name, "sentiment")

        if not model or not tokenizer:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        # Process the request
        texts = request.text if isinstance(request.text, list) else [request.text]

        # Check if aspect-based sentiment analysis is requested
        if request.aspects:
            # Use aspect-based sentiment model if available
            aspect_model = get_model(model_name + "_absa", "sentiment")
            if aspect_model:
                sentiments = aspect_model.analyze(
                    texts=texts, aspects=request.aspects, tokenizer=tokenizer
                )

                return SentimentResponse(
                    sentiments=sentiments, model_name=model_name + "_absa", probabilities=None
                )
            else:
                # Fall back to regular sentiment analysis
                logger.warning(
                    f"Aspect-based model not found for {model_name}, falling back to standard model"
                )

        # Standard sentiment analysis
        if request.return_probabilities:
            # Tokenize
            inputs = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )

            # Get predictions with probabilities
            probabilities = model.predict(inputs, return_probabilities=True)

            # Get labels from model
            if hasattr(model, "id2label"):
                labels = [model.id2label[i] for i in range(len(model.id2label))]
            else:
                # Default labels
                labels = ["negative", "neutral", "positive"]

            # Format probabilities as dicts
            formatted_probs = []
            for prob in probabilities:
                prob_dict = {label: float(p) for label, p in zip(labels, prob)}
                formatted_probs.append(prob_dict)

            # Get most likely sentiment
            predictions = [labels[prob.argmax()] for prob in probabilities]

            return SentimentResponse(
                sentiments=predictions,
                probabilities=formatted_probs,
                model_name=model_name,
            )
        else:
            # Direct prediction without probabilities
            predictions = model.predict_text(texts, tokenizer=tokenizer)

            return SentimentResponse(
                sentiments=predictions, model_name=model_name, probabilities=None
            )

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Summarization routes
@summarization_router.post("/predict", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """Summarize text."""
    try:
        # Get model and tokenizer based on method
        if request.method == "abstractive":
            model_name = request.model_name or "default_summarization"
            model = get_model(model_name, "summarization")
            tokenizer = get_tokenizer(model_name, "summarization")
        else:  # extractive
            model_name = "extractive_" + (request.model_name or "tfidf")
            model = get_model(model_name, "summarization")
            tokenizer = None  # Extractive doesn't need a tokenizer

        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        # Process the request
        texts = request.text if isinstance(request.text, list) else [request.text]

        # Generate summaries
        if request.method == "abstractive":
            # Set length parameters based on preset
            if request.length == "short":
                max_length = 64
                min_length = 10
            elif request.length == "medium":
                max_length = 128
                min_length = 30
            else:  # long
                max_length = 256
                min_length = 64

            # Override with explicit values if provided
            max_length = request.max_length or max_length
            min_length = request.min_length or min_length

            # Check if controlled summarization is requested
            if request.focus:
                # Get controlled summarizer if available
                ctrl_model = get_model(model_name + "_controlled", "summarization")
                if ctrl_model:
                    summaries = [
                        ctrl_model.summarize(
                            text=text,
                            tokenizer=tokenizer,
                            length=request.length,
                            focus=request.focus,
                        )
                        for text in texts
                    ]

                    return SummarizationResponse(
                        summaries=summaries, model_name=model_name + "_controlled"
                    )

            # Regular abstractive summarization
            summaries = model.summarize_text(
                texts=texts,
                tokenizer=tokenizer,
                max_length=max_length,
                min_length=min_length,
            )
        else:
            # Extractive summarization
            # Calculate ratio based on length
            if request.length == "short":
                ratio = 0.2
            elif request.length == "medium":
                ratio = 0.3
            else:  # long
                ratio = 0.4

            # Generate extractive summaries
            summaries = [model.summarize(text, ratio=ratio) for text in texts]

        return SummarizationResponse(summaries=summaries, model_name=model_name)

    except Exception as e:
        logger.error(f"Error in summarization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Model Hub routes
@model_hub_router.get("/list", response_model=List[ModelInfo])
async def list_models(task: Optional[str] = None):
    """List available models, optionally filtered by task."""
    try:
        # Get models from registry
        models = list_available_models(task)

        return models

    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@model_hub_router.get("/info/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        # Find model in registry
        models = list_available_models()

        for model in models:
            if model.name == model_name:
                return model

        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
