"""
Text classification models and utilities.

This module provides classes and functions for text classification tasks.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)


class TransformerClassifier:
    """
    Wrapper class for transformer-based text classification models.

    This class provides a unified interface for working with different
    transformer models for classification tasks.
    """

    def __init__(
        self,
        model_name_or_path: str = None,
        num_labels: int = 2,
        problem_type: Optional[str] = None,
        label_map: Optional[Dict[int, str]] = None,
        device: Optional[str] = None,
        model_name: str = None,  # Added for backward compatibility
    ):
        """
        Initialize the classifier with a pre-trained model.

        Args:
            model_name_or_path: Hugging Face model name or path to local model
            num_labels: Number of classification labels
            problem_type: Problem type ('single_label_classification', 'multi_label_classification')
            label_map: Mapping from label IDs to label names
            device: Device to use ('cpu', 'cuda', 'mps')
            model_name: Alternative parameter name for model_name_or_path (for backward compatibility)
        """
        # Handle parameter naming flexibility (for notebook compatibility)
        if model_name_or_path is None and model_name is not None:
            model_name_or_path = model_name
        elif model_name_or_path is None and model_name is None:
            raise ValueError("Either model_name_or_path or model_name must be provided")
            
        self.model_name = model_name_or_path
        self.num_labels = num_labels
        self.label_map = label_map

        # Set problem type
        if problem_type is None:
            if num_labels == 2:
                problem_type = "single_label_classification"
            elif num_labels > 2:
                problem_type = "single_label_classification"
                # Could be multi-label too, but we default to multi-class
        self.problem_type = problem_type

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load configuration and model
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels, problem_type=problem_type
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )

        self.model.to(self.device)

        # Load tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def train(
        self,
        train_dataloader,
        eval_dataloader=None,
        val_dataloader=None,
        optimizer=None,
        scheduler=None,
        num_epochs: int = 3,
        max_grad_norm: float = 1.0,
        eval_steps: int = 100,
        save_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Train the classifier model.

        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            val_dataloader: Alternative name for eval_dataloader
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of training epochs
            max_grad_norm: Maximum gradient norm for gradient clipping
            eval_steps: Number of steps between evaluations
            save_path: Path to save the model
            output_dir: Alternative name for save_path

        Returns:
            Training history (losses, metrics)
        """
        # Handle parameter aliases
        if eval_dataloader is None and val_dataloader is not None:
            eval_dataloader = val_dataloader
            
        if save_path is None and output_dir is not None:
            save_path = output_dir
            
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

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
        Evaluate the classification model.

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
                if self.problem_type == "multi_label_classification":
                    preds = torch.sigmoid(logits) > 0.5
                else:  # single_label_classification
                    preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate metrics
        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": (torch.tensor(all_preds) == torch.tensor(all_labels))
            .float()
            .mean()
            .item(),
        }

        return metrics

    def predict(
        self,
        encoded_inputs,
        return_probabilities=False,
    ):
        """
        Make predictions with the classification model.

        Args:
            encoded_inputs: Encoded inputs (from tokenizer) or raw text input (str or list of str)
            return_probabilities: Whether to return probabilities or labels

        Returns:
            Predictions (labels or probabilities)
        """
        self.model.eval()

        # Handle raw text input
        is_raw_text = isinstance(encoded_inputs, str) or (
            isinstance(encoded_inputs, list) and all(isinstance(item, str) for item in encoded_inputs)
        )
        
        if is_raw_text:
            # Convert single string to list for batch processing
            if isinstance(encoded_inputs, str):
                encoded_inputs = [encoded_inputs]
                
            # Tokenize the text inputs
            inputs = self.tokenizer(
                encoded_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        else:
            # Use the pre-encoded inputs
            inputs = encoded_inputs
            
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            if return_probabilities:
                if self.problem_type == "multi_label_classification":
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:  # single_label_classification
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                return probs
            else:
                if self.problem_type == "multi_label_classification":
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                else:  # single_label_classification
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()

                # Map numeric labels to strings if label_map is provided
                if self.label_map is not None:
                    preds = [self.label_map[pred] for pred in preds]

                return preds

    def get_model_size(self):
        """
        Get the number of parameters in the model.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())

    def save(self, path: str):
        """
        Save the model and its configuration.

        Args:
            path: Directory path to save the model
        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.model.save_pretrained(path)

        # Save label map if exists
        if self.label_map is not None:
            import json

            with open(os.path.join(path, "label_map.json"), "w") as f:
                json.dump(self.label_map, f)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load a model from a directory.

        Args:
            path: Directory path containing the saved model
            device: Device to load the model on

        Returns:
            TransformerClassifier instance
        """
        # Load config to get num_labels and problem_type
        config = AutoConfig.from_pretrained(path)

        # Load label map if exists
        label_map = None
        label_map_path = os.path.join(path, "label_map.json")
        if os.path.exists(label_map_path):
            import json

            with open(label_map_path, "r") as f:
                label_map = json.load(f)
                # Convert string keys back to integers
                label_map = {int(k): v for k, v in label_map.items()}

        # Create instance
        instance = cls(
            model_name_or_path=path,
            num_labels=config.num_labels,
            problem_type=config.problem_type,
            label_map=label_map,
            device=device,
        )

        return instance


class EnsembleClassifier:
    """
    Ensemble of transformer classifiers for better performance.

    This class combines multiple transformer models for text classification.
    """

    def __init__(
        self,
        model_names: List[str],
        num_labels: int = 2,
        problem_type: Optional[str] = None,
        label_map: Optional[Dict[int, str]] = None,
        device: Optional[str] = None,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize an ensemble of classifiers.

        Args:
            model_names: List of model names or paths
            num_labels: Number of classification labels
            problem_type: Problem type
            label_map: Mapping from label IDs to label names
            device: Device to use
            weights: Weights for each model in the ensemble (default: equal weights)
        """
        self.model_names = model_names
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.label_map = label_map

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Create individual classifiers
        self.classifiers = []
        for model_name in model_names:
            classifier = TransformerClassifier(
                model_name_or_path=model_name,
                num_labels=num_labels,
                problem_type=problem_type,
                label_map=label_map,
                device=device,
            )
            self.classifiers.append(classifier)

        # Set weights (default: equal weights)
        if weights is None:
            self.weights = [1.0 / len(self.classifiers)] * len(self.classifiers)
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(
        self,
        encoded_inputs,
        return_probabilities=False,
    ):
        """
        Make predictions with the ensemble.

        Args:
            encoded_inputs: Encoded inputs (from tokenizer) or raw text input (str or list of str)
            return_probabilities: Whether to return probabilities or labels

        Returns:
            Predictions (labels or probabilities)
        """
        # Get predictions from each model
        all_probs = []

        for classifier in self.classifiers:
            probs = classifier.predict(encoded_inputs, return_probabilities=True)
            all_probs.append(probs)

        # Compute weighted average of probabilities
        ensemble_probs = sum(w * p for w, p in zip(self.weights, all_probs))

        if return_probabilities:
            return ensemble_probs
        else:
            if self.problem_type == "multi_label_classification":
                preds = (ensemble_probs > 0.5).astype(int)
            else:  # single_label_classification
                preds = ensemble_probs.argmax(axis=-1)

            # Map numeric labels to strings if label_map is provided
            if self.label_map is not None:
                preds = [self.label_map[pred] for pred in preds]

            return preds

    def save(self, path: str):
        """
        Save all models in the ensemble.

        Args:
            path: Directory path to save the ensemble
        """
        if not os.path.exists(path):
            os.makedirs(path)

        # Save each model in a subdirectory
        for i, classifier in enumerate(self.classifiers):
            model_path = os.path.join(path, f"model_{i}")
            classifier.save(model_path)

        # Save weights and model names
        import json

        ensemble_config = {
            "model_names": self.model_names,
            "weights": self.weights,
            "num_labels": self.num_labels,
            "problem_type": self.problem_type,
        }

        with open(os.path.join(path, "ensemble_config.json"), "w") as f:
            json.dump(ensemble_config, f)

        # Save label map if exists
        if self.label_map is not None:
            with open(os.path.join(path, "label_map.json"), "w") as f:
                json.dump(self.label_map, f)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load an ensemble from a directory.

        Args:
            path: Directory path containing the saved ensemble
            device: Device to load the models on

        Returns:
            EnsembleClassifier instance
        """
        import json

        # Load ensemble configuration
        with open(os.path.join(path, "ensemble_config.json"), "r") as f:
            config = json.load(f)

        # Load label map if exists
        label_map = None
        label_map_path = os.path.join(path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                label_map = json.load(f)
                # Convert string keys back to integers
                label_map = {int(k): v for k, v in label_map.items()}

        # Get number of models from directory structure
        model_paths = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d.startswith("model_")
        ]

        # Create instance with correct params but don't load models yet
        instance = cls(
            model_names=[],  # Empty for now
            num_labels=config["num_labels"],
            problem_type=config["problem_type"],
            label_map=label_map,
            device=device,
            weights=config["weights"],
        )

        # Replace the classifiers list with loaded models
        instance.classifiers = []
        for model_path in sorted(model_paths):
            classifier = TransformerClassifier.load(model_path, device=device)
            instance.classifiers.append(classifier)

        # Set model names from config
        instance.model_names = config["model_names"]

        return instance


# Factory function for creating classifiers with specific architectures
def create_classifier(model_name: str, num_labels: int, **kwargs):
    """
    Create a transformer classifier based on model name.

    Args:
        model_name: Model name or path
        num_labels: Number of labels
        **kwargs: Additional arguments for TransformerClassifier

    Returns:
        TransformerClassifier instance
    """
    return TransformerClassifier(
        model_name_or_path=model_name, num_labels=num_labels, **kwargs
    )


def create_ensemble(model_names: List[str], num_labels: int, **kwargs):
    """
    Create an ensemble classifier.

    Args:
        model_names: List of model names or paths
        num_labels: Number of labels
        **kwargs: Additional arguments for EnsembleClassifier

    Returns:
        EnsembleClassifier instance
    """
    return EnsembleClassifier(model_names=model_names, num_labels=num_labels, **kwargs)
