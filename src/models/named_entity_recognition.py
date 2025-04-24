"""
Named Entity Recognition (NER) models and utilities.

This module provides classes and functions for training, evaluating
and using transformer-based models for named entity recognition.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoConfig, AutoModelForTokenClassification, PreTrainedModel


class NERModel:
    """
    Wrapper class for transformer-based named entity recognition models.

    This class provides a unified interface for working with different
    transformer models for token classification tasks like NER.
    """

    def __init__(
        self,
        model_name_or_path: str = None,
        num_labels: int = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        device: Optional[str] = None,
        model_name: str = None,  # Added for backward compatibility
    ):
        """
        Initialize the NER model with a pre-trained model.

        Args:
            model_name_or_path: Hugging Face model name or path to local model
            num_labels: Number of NER label types (if None, will attempt to detect from model config)
            id2label: Mapping from ID to label name
            label2id: Mapping from label name to ID
            device: Device to use ('cpu', 'cuda', 'mps')
            model_name: Alternative parameter name for model_name_or_path (for backward compatibility)
        """
        # Handle parameter naming flexibility (for notebook compatibility)
        if model_name_or_path is None and model_name is not None:
            model_name_or_path = model_name
        elif model_name_or_path is None and model_name is None:
            raise ValueError("Either model_name_or_path or model_name must be provided")
            
        self.model_name = model_name_or_path
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load configuration first to detect num_labels if not provided
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        # Use num_labels from config if not explicitly provided
        if num_labels is None:
            if hasattr(config, 'num_labels'):
                num_labels = config.num_labels
            else:
                # Default to 9 labels for standard NER (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC)
                num_labels = 9
        
        self.num_labels = num_labels
        
        # Get or set label mappings
        if id2label is None and hasattr(config, 'id2label'):
            id2label = config.id2label
        self.id2label = id2label
        
        if label2id is None and hasattr(config, 'label2id'):
            label2id = config.label2id
        self.label2id = label2id

        # Load the rest of the model
        if self.id2label is not None and self.label2id is not None:
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        else:
            self.config = config
            # Update config with our num_labels if needed
            if self.config.num_labels != self.num_labels:
                self.config.num_labels = self.num_labels

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, config=self.config
        )

        self.model.to(self.device)

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        eval_dataloader=None,
        optimizer=None,
        scheduler=None,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        eval_steps: int = 100,
        save_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_best: bool = False,
    ):
        """
        Train the NER model.

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (alias for eval_dataloader)
            eval_dataloader: DataLoader for evaluation data
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for gradient clipping
            eval_steps: Number of steps between evaluations
            save_path: Path to save the model
            output_dir: Directory to save the model (alias for save_path)
            save_best: Whether to save the best model based on validation loss

        Returns:
            Training history (losses, metrics)
        """
        # Handle parameter aliases for compatibility
        if val_dataloader is not None and eval_dataloader is None:
            eval_dataloader = val_dataloader
            
        if output_dir is not None and save_path is None:
            save_path = output_dir
            
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )

        history = {"train_loss": [], "eval_loss": [], "f1_score": []}
        best_f1 = 0.0

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
                    history["f1_score"].append(eval_results["f1_score"])
                    
                    # Save best model if requested
                    if save_best and save_path and eval_results["f1_score"] > best_f1:
                        best_f1 = eval_results["f1_score"]
                        best_model_path = os.path.join(save_path, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        self.save(best_model_path)
                        print(f"Saved best model with F1 score: {best_f1:.4f}")

                    # Print progress
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | Step {step} | "
                        f"Train Loss: {loss.item():.4f} | "
                        f"Eval Loss: {eval_results['loss']:.4f} | "
                        f"F1 Score: {eval_results['f1_score']:.4f}"
                    )

                    # Return to training mode
                    self.model.train()

            # Print epoch summary
            print(
                f"Epoch {epoch+1}/{num_epochs} completed | "
                f"Average Loss: {epoch_loss/len(train_dataloader):.4f}"
            )

        # Save the final model if path provided
        if save_path:
            final_model_path = os.path.join(save_path, "final_model") if save_best else save_path
            os.makedirs(final_model_path, exist_ok=True)
            self.save(final_model_path)

        return history

    def evaluate(self, dataloader):
        """
        Evaluate the NER model.

        Args:
            dataloader: DataLoader for evaluation data

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        true_predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)

                # We need to filter out padded tokens
                for i, label in enumerate(batch["labels"]):
                    true_label = []
                    true_prediction = []

                    for j, lbl in enumerate(label):
                        if lbl != -100:  # -100 is the padding token in HF datasets
                            true_label.append(self.id2label[lbl.item()])
                            true_prediction.append(
                                self.id2label[predictions[i, j].item()]
                            )

                    true_labels.append(true_label)
                    true_predictions.append(true_prediction)

        # Calculate metrics
        metrics = {
            "loss": total_loss / len(dataloader),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1_score": f1_score(true_labels, true_predictions),
        }

        return metrics

    def predict(
        self,
        inputs,
        tokenizer=None,
        original_texts: Optional[List[str]] = None,
        align_to_words: bool = True,
    ):
        """
        Make predictions with the NER model.

        Args:
            inputs: Either encoded inputs dict or raw text (string or list of strings)
            tokenizer: Tokenizer to use if raw text is provided (optional if inputs are encoded)
            original_texts: Original texts corresponding to the inputs (only needed for encoded inputs)
            align_to_words: Whether to align predictions to words in original text

        Returns:
            Predictions as list of entity dictionaries for each text
        """
        self.model.eval()
        
        # Handle different input types
        if isinstance(inputs, str):
            # Single text string
            texts = [inputs]
            encoded_inputs = None
        elif isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
            # List of text strings
            texts = inputs
            encoded_inputs = None
        else:
            # Assume it's already encoded inputs
            encoded_inputs = inputs
            texts = original_texts if original_texts else []
            
        # Tokenize raw text if needed
        if encoded_inputs is None:
            if tokenizer is None:
                # Load tokenizer if not provided
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
            # Tokenize the input texts
            encoded_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=align_to_words,
                max_length=512
            )
            
            # Set original texts for entity alignment
            original_texts = texts

        # Process encoded inputs
        device_inputs = {}
        offset_mapping = None
        
        for k, v in encoded_inputs.items():
            if k == "offset_mapping":
                offset_mapping = v
            else:
                device_inputs[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model(**device_inputs)
            logits = outputs.logits

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

        batch_results = []

        # Process each sequence in the batch
        for i, pred in enumerate(predictions):
            sequence_pred = pred.cpu().numpy()
            
            if align_to_words and tokenizer and original_texts:
                # Get word-aligned entities
                entities = self._align_predictions_to_words(
                    sequence_pred,
                    original_texts[i],
                    offset_mapping[i] if offset_mapping is not None else None,
                    tokenizer
                )
                batch_results.append(entities)
            else:
                # Just return raw predictions for each token
                entities = []
                current_entity = None
                
                for j, label_id in enumerate(sequence_pred):
                    if label_id == 0:  # 'O' label
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None
                    else:
                        label = self.id2label[label_id] if self.id2label else str(label_id)
                        if label.startswith("B-"):
                            if current_entity:
                                entities.append(current_entity)
                            current_entity = {
                                "entity": label[2:],
                                "token_index": j,
                                "score": 1.0  # We don't have token-level scores without softmax
                            }
                        elif label.startswith("I-") and current_entity:
                            # Extend current entity
                            pass
                
                if current_entity:
                    entities.append(current_entity)
                    
                batch_results.append(entities)

        return batch_results
        
    def _align_predictions_to_words(self, token_predictions, text, offset_mapping=None, tokenizer=None):
        """
        Align token predictions to words in the original text.
        
        Args:
            token_predictions: Predicted label IDs for each token
            text: Original text
            offset_mapping: Mapping from tokens to character positions
            tokenizer: Tokenizer used for encoding
            
        Returns:
            List of entity dictionaries with start/end positions and entity types
        """
        entities = []
        current_entity = None
        
        if offset_mapping is None:
            # Without offset mapping, we can't do proper alignment
            return []
            
        for i, (offset, label_id) in enumerate(zip(offset_mapping, token_predictions)):
            # Skip special tokens with offset (0, 0)
            if offset[0] == offset[1] == 0:
                continue
                
            # Get the label
            if self.id2label:
                label = self.id2label[int(label_id)]
            else:
                label = str(label_id)
                
            # For 'O' label, finalize any current entity
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            else:
                # For named entities
                entity_type = label[2:]  # Remove B- or I- prefix
                
                if label.startswith("B-"):
                    # Start of a new entity
                    if current_entity:
                        entities.append(current_entity)
                        
                    current_entity = {
                        "start": int(offset[0]),
                        "end": int(offset[1]),
                        "entity": entity_type,
                        "score": 1.0  # We don't have proper scores without softmax probabilities
                    }
                elif label.startswith("I-") and current_entity and current_entity["entity"] == entity_type:
                    # Continuation of the current entity
                    current_entity["end"] = int(offset[1])
        
        # Add the last entity if we had one in progress
        if current_entity:
            entities.append(current_entity)
            
        return entities

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

        # Save id2label and label2id mappings
        if self.id2label is not None and self.label2id is not None:
            import json

            with open(os.path.join(path, "label_mappings.json"), "w") as f:
                json.dump(
                    {
                        "id2label": {str(k): v for k, v in self.id2label.items()},
                        "label2id": self.label2id,
                    },
                    f,
                )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load a model from a directory.

        Args:
            path: Directory path containing the saved model
            device: Device to load the model on

        Returns:
            NERModel instance
        """
        # Load config to get num_labels
        config = AutoConfig.from_pretrained(path)

        # Load label mappings if they exist
        id2label = config.id2label
        label2id = config.label2id

        label_mappings_path = os.path.join(path, "label_mappings.json")
        if os.path.exists(label_mappings_path):
            import json

            with open(label_mappings_path, "r") as f:
                mappings = json.load(f)
                id2label = {int(k): v for k, v in mappings["id2label"].items()}
                label2id = mappings["label2id"]

        # Create instance
        instance = cls(
            model_name_or_path=path,
            num_labels=config.num_labels,
            id2label=id2label,
            label2id=label2id,
            device=device,
        )

        return instance


class CustomNERModel(nn.Module):
    """
    Custom NER model with additional features beyond standard transformers.

    This class implements a token classification model with:
    - Transformer backbone
    - Additional features (e.g., character-level features)
    - CRF layer for better sequence modeling
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        use_crf: bool = True,
        use_char_features: bool = False,
        char_vocab_size: Optional[int] = None,
        char_embedding_dim: int = 30,
        char_hidden_size: int = 50,
        dropout: float = 0.1,
    ):
        """
        Initialize custom NER model.

        Args:
            model_name_or_path: Transformer model name or path
            num_labels: Number of NER label types
            use_crf: Whether to use a CRF layer
            use_char_features: Whether to use character-level features
            char_vocab_size: Size of character vocabulary
            char_embedding_dim: Dimension of character embeddings
            char_hidden_size: Size of character-level RNN hidden state
            dropout: Dropout probability
        """
        super().__init__()

        # Transformer backbone
        self.transformer = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=(
                num_labels if not use_crf else None
            ),  # No classification head if using CRF
        )

        # Get hidden size from transformer
        hidden_size = self.transformer.config.hidden_size

        # Character-level features
        self.use_char_features = use_char_features
        if use_char_features:
            assert (
                char_vocab_size is not None
            ), "char_vocab_size must be specified for character features"

            self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
            self.char_lstm = nn.LSTM(
                char_embedding_dim,
                char_hidden_size,
                batch_first=True,
                bidirectional=True,
            )

            # Update hidden size to include character features
            hidden_size += char_hidden_size * 2  # Bidirectional

        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # CRF layer
        self.use_crf = use_crf
        if use_crf:
            # Placeholder for CRF implementation
            # In a real implementation, you'd use a CRF library like TorchCRF
            pass

    def forward(
        self, input_ids, attention_mask, token_type_ids=None, char_ids=None, labels=None
    ):
        """
        Forward pass of the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            char_ids: Character IDs (if using character features)
            labels: Labels for training

        Returns:
            Model outputs
        """
        # Get transformer outputs
        transformer_outputs = self.transformer.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = transformer_outputs[0]

        # Add character-level features if enabled
        if self.use_char_features and char_ids is not None:
            batch_size, seq_length, char_length = char_ids.size()

            # Reshape for character LSTM
            char_ids = char_ids.view(-1, char_length)
            char_embeddings = self.char_embedding(char_ids)

            # Run character LSTM
            char_lstm_output, _ = self.char_lstm(char_embeddings)

            # Get last hidden state
            char_features = char_lstm_output[:, -1, :]

            # Reshape back to batch format
            char_features = char_features.view(batch_size, seq_length, -1)

            # Concatenate with transformer features
            sequence_output = torch.cat([sequence_output, char_features], dim=-1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.use_crf:
                # CRF loss implementation would go here
                pass
            else:
                # Standard cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                # Only consider active parts of the loss
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)

        return {"loss": loss, "logits": logits}


# Factory functions
def create_ner_model(model_name: str, num_labels: int, **kwargs):
    """
    Create a NER model based on model name.

    Args:
        model_name: Model name or path
        num_labels: Number of NER label types
        **kwargs: Additional arguments for NERModel

    Returns:
        NERModel instance
    """
    return NERModel(model_name_or_path=model_name, num_labels=num_labels, **kwargs)


def create_custom_ner_model(model_name: str, num_labels: int, **kwargs):
    """
    Create a custom NER model with additional features.

    Args:
        model_name: Model name or path
        num_labels: Number of NER label types
        **kwargs: Additional arguments for CustomNERModel

    Returns:
        CustomNERModel instance
    """
    return CustomNERModel(
        model_name_or_path=model_name, num_labels=num_labels, **kwargs
    )
