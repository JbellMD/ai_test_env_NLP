"""
Sentiment analysis models and utilities.

This module provides classes and functions for sentiment analysis,
including fine-grained and aspect-based sentiment analysis.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class SentimentAnalyzer:
    """
    Wrapper class for transformer-based sentiment analysis models.

    This class provides a unified interface for sentiment analysis,
    supporting both binary and fine-grained sentiment analysis.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 2,
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the sentiment analyzer with a pre-trained model.

        Args:
            model_name_or_path: Hugging Face model name or path to local model
            num_labels: Number of sentiment labels
            labels: List of sentiment labels (e.g., ["negative", "neutral", "positive"])
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name_or_path
        self.num_labels = num_labels

        # Set up label mapping
        if labels is None:
            if num_labels == 2:
                self.labels = ["negative", "positive"]
            elif num_labels == 3:
                self.labels = ["negative", "neutral", "positive"]
            elif num_labels == 5:
                self.labels = [
                    "very negative",
                    "negative",
                    "neutral",
                    "positive",
                    "very positive",
                ]
            else:
                self.labels = [str(i) for i in range(num_labels)]
        else:
            self.labels = labels

        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load configuration and model
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )

        self.model.to(self.device)

    def train(
        self,
        train_dataloader,
        eval_dataloader=None,
        optimizer=None,
        scheduler=None,
        num_epochs: int = 3,
        max_grad_norm: float = 1.0,
        eval_steps: int = 100,
        save_path: Optional[str] = None,
    ):
        """
        Train the sentiment analysis model.

        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of training epochs
            max_grad_norm: Maximum gradient norm for gradient clipping
            eval_steps: Number of steps between evaluations
            save_path: Path to save the model

        Returns:
            Training history (losses, metrics)
        """
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        history = {"train_loss": [], "eval_loss": [], "accuracy": []}

        self.model.train()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0

            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Update LR scheduler if provided
                if scheduler is not None:
                    scheduler.step()

                epoch_loss += loss.item()
                history["train_loss"].append(loss.item())

                global_step += 1

                # Evaluate if needed
                if eval_dataloader is not None and global_step % eval_steps == 0:
                    eval_results = self.evaluate(eval_dataloader)
                    history["eval_loss"].append(eval_results["loss"])
                    history["accuracy"].append(eval_results["accuracy"])

                    # Print progress
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | Step {step} | "
                        f"Train Loss: {loss.item():.4f} | "
                        f"Eval Loss: {eval_results['loss']:.4f} | "
                        f"Accuracy: {eval_results['accuracy']:.4f}"
                    )

                    # Return to training mode
                    self.model.train()

            # Print epoch summary
            print(
                f"Epoch {epoch+1}/{num_epochs} completed | "
                f"Average Loss: {epoch_loss/len(train_dataloader):.4f}"
            )

        # Save the model if path provided
        if save_path:
            self.save(save_path)

        return history

    def evaluate(self, dataloader):
        """
        Evaluate the sentiment analysis model.

        Args:
            dataloader: DataLoader for evaluation data

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                # Get predictions
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate metrics
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

        # Calculate per-class accuracy and F1 score
        from sklearn.metrics import classification_report

        report = classification_report(
            all_labels, all_preds, target_names=self.labels, output_dict=True
        )

        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": accuracy,
            "per_class_metrics": report,
        }

        return metrics

    def predict(
        self,
        encoded_inputs: Dict[str, torch.Tensor],
        return_probabilities: bool = False,
    ):
        """
        Predict sentiment for input text.

        Args:
            encoded_inputs: Encoded inputs from tokenizer
            return_probabilities: Whether to return probabilities or labels

        Returns:
            Sentiment predictions (labels or probabilities)
        """
        self.model.eval()

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            if return_probabilities:
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                return probs
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                # Map numeric predictions to label strings
                pred_labels = [self.id2label[pred] for pred in preds]
                return pred_labels

    def predict_text(
        self,
        texts: Union[str, List[str]],
        tokenizer,
        return_probabilities: bool = False,
        batch_size: int = 8,
    ):
        """
        Predict sentiment for raw text input.

        Args:
            texts: Input text or list of texts
            tokenizer: Tokenizer to use
            return_probabilities: Whether to return probabilities or labels
            batch_size: Batch size for processing

        Returns:
            Sentiment predictions
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        all_predictions = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            )

            # Get predictions
            batch_predictions = self.predict(
                inputs, return_probabilities=return_probabilities
            )
            all_predictions.extend(batch_predictions)

        # If input was a single text, return a single prediction
        if len(texts) == 1 and isinstance(texts, list):
            return all_predictions[0]

        return all_predictions

    def save(self, path: str):
        """
        Save the model and its configuration.

        Args:
            path: Directory path to save the model
        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.model.save_pretrained(path)

        # Save labels
        import json

        with open(os.path.join(path, "sentiment_labels.json"), "w") as f:
            json.dump(self.labels, f)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load a model from a directory.

        Args:
            path: Directory path containing the saved model
            device: Device to load the model on

        Returns:
            SentimentAnalyzer instance
        """
        # Load config to get num_labels
        config = AutoConfig.from_pretrained(path)

        # Load labels if exists
        labels_path = os.path.join(path, "sentiment_labels.json")
        if os.path.exists(labels_path):
            import json

            with open(labels_path, "r") as f:
                labels = json.load(f)
        else:
            labels = None

        # Create instance
        instance = cls(
            model_name_or_path=path,
            num_labels=config.num_labels,
            labels=labels,
            device=device,
        )

        return instance


class AspectBasedSentimentAnalyzer:
    """
    Aspect-Based Sentiment Analysis (ABSA) model.

    This class handles sentiment analysis for specific aspects or
    attributes mentioned in the text.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 3,
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the ABSA model.

        Args:
            model_name_or_path: Hugging Face model name or path to local model
            num_labels: Number of sentiment labels
            labels: List of sentiment labels
            device: Device to use
        """
        self.sentiment_analyzer = SentimentAnalyzer(
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
            labels=labels,
            device=device,
        )

        self.device = self.sentiment_analyzer.device

    def extract_aspects(self, texts: Union[str, List[str]], ner_model=None):
        """
        Extract aspects from text using NER or other extraction methods.

        Args:
            texts: Input text or list of texts
            ner_model: Named Entity Recognition model for aspect extraction

        Returns:
            List of extracted aspects for each text
        """
        # This is a placeholder for actual aspect extraction logic
        if ner_model is not None:
            # Use the NER model to extract aspects
            # Implementation would depend on the specific NER model
            pass

        # Placeholder implementation: split text by commas or 'and'
        if isinstance(texts, str):
            texts = [texts]

        all_aspects = []

        for text in texts:
            # Simple rule-based aspect extraction (for demonstration)
            # In a real implementation, you would use NER or dependency parsing
            aspects = []
            sentences = text.split(".")

            for sentence in sentences:
                words = sentence.strip().split()

                # Simple rule: nouns preceded by adjectives might be aspects
                for i in range(1, len(words)):
                    # This is a very naive approach - real implementation would be more sophisticated
                    if len(words[i]) > 3:  # Simple heuristic to identify nouns
                        aspects.append(words[i])

            all_aspects.append(aspects)

        return all_aspects

    def analyze(
        self,
        texts: Union[str, List[str]],
        aspects: Optional[Union[List[str], List[List[str]]]] = None,
        tokenizer=None,
        extract_aspects: bool = True,
    ):
        """
        Perform aspect-based sentiment analysis.

        Args:
            texts: Input text or list of texts
            aspects: Predefined aspects to analyze, or None to extract automatically
            tokenizer: Tokenizer to use
            extract_aspects: Whether to extract aspects automatically if not provided

        Returns:
            Dictionary mapping aspects to sentiment for each text
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Extract aspects if needed
        if aspects is None and extract_aspects:
            aspects = self.extract_aspects(texts)
        elif aspects is not None and isinstance(aspects[0], str):
            # If aspects is a list of strings, use the same aspects for all texts
            aspects = [aspects] * len(texts)

        results = []

        for i, text in enumerate(texts):
            aspect_sentiments = {}

            text_aspects = aspects[i] if aspects is not None else []

            for aspect in text_aspects:
                # For each aspect, create an aspect-focused input
                # This is a simple approach - more sophisticated methods exist
                aspect_text = f"How is the {aspect} in: {text}"

                # Get sentiment for this aspect
                sentiment = self.sentiment_analyzer.predict_text(
                    aspect_text, tokenizer=tokenizer, return_probabilities=False
                )

                aspect_sentiments[aspect] = sentiment

            results.append(aspect_sentiments)

        # If input was a single text, return a single result
        if single_input:
            return results[0]

        return results


# Factory functions
def create_sentiment_analyzer(model_name: str, num_labels: int = 3, **kwargs):
    """
    Create a sentiment analyzer based on model name.

    Args:
        model_name: Model name or path
        num_labels: Number of sentiment labels
        **kwargs: Additional arguments for SentimentAnalyzer

    Returns:
        SentimentAnalyzer instance
    """
    return SentimentAnalyzer(
        model_name_or_path=model_name, num_labels=num_labels, **kwargs
    )


def create_absa_analyzer(model_name: str, num_labels: int = 3, **kwargs):
    """
    Create an aspect-based sentiment analyzer.

    Args:
        model_name: Model name or path
        num_labels: Number of sentiment labels
        **kwargs: Additional arguments for AspectBasedSentimentAnalyzer

    Returns:
        AspectBasedSentimentAnalyzer instance
    """
    return AspectBasedSentimentAnalyzer(
        model_name_or_path=model_name, num_labels=num_labels, **kwargs
    )
