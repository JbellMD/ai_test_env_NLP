"""
Named Entity Recognition (NER) models and utilities.

This module provides classes and functions for training, evaluating
and using transformer-based models for named entity recognition.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    PreTrainedModel
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from typing import Dict, List, Optional, Union, Any, Tuple


class NERModel:
    """
    Wrapper class for transformer-based named entity recognition models.
    
    This class provides a unified interface for working with different
    transformer models for token classification tasks like NER.
    """
    
    def __init__(self, 
                 model_name_or_path: str,
                 num_labels: int,
                 id2label: Optional[Dict[int, str]] = None,
                 label2id: Optional[Dict[str, int]] = None,
                 device: Optional[str] = None):
        """
        Initialize the NER model with a pre-trained model.
        
        Args:
            model_name_or_path: Hugging Face model name or path to local model
            num_labels: Number of NER label types
            id2label: Mapping from ID to label name
            label2id: Mapping from label name to ID
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name_or_path
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load configuration and model
        if id2label is not None and label2id is not None:
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )
        else:
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=num_labels
            )
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            config=self.config
        )
        
        self.model.to(self.device)
    
    def train(self, 
              train_dataloader, 
              eval_dataloader=None, 
              optimizer=None,
              scheduler=None,
              num_epochs: int = 3,
              max_grad_norm: float = 1.0,
              eval_steps: int = 100,
              save_path: Optional[str] = None):
        """
        Train the NER model.
        
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
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        history = {
            'train_loss': [],
            'eval_loss': [],
            'f1_score': []
        }
        
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
                history['train_loss'].append(loss.item())
                
                global_step += 1
                
                # Evaluate if needed
                if eval_dataloader is not None and global_step % eval_steps == 0:
                    eval_results = self.evaluate(eval_dataloader)
                    history['eval_loss'].append(eval_results['loss'])
                    history['f1_score'].append(eval_results['f1_score'])
                    
                    # Print progress
                    print(f"Epoch {epoch+1}/{num_epochs} | Step {step} | "
                          f"Train Loss: {loss.item():.4f} | "
                          f"Eval Loss: {eval_results['loss']:.4f} | "
                          f"F1 Score: {eval_results['f1_score']:.4f}")
                    
                    # Return to training mode
                    self.model.train()
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} completed | "
                  f"Average Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        # Save the model if path provided
        if save_path:
            self.save(save_path)
        
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
                            true_prediction.append(self.id2label[predictions[i, j].item()])
                    
                    true_labels.append(true_label)
                    true_predictions.append(true_prediction)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(dataloader),
            'precision': precision_score(true_labels, true_predictions),
            'recall': recall_score(true_labels, true_predictions),
            'f1_score': f1_score(true_labels, true_predictions)
        }
        
        return metrics
    
    def predict(self, 
               encoded_inputs: Dict[str, torch.Tensor], 
               tokenizer=None,
               original_texts: Optional[List[str]] = None,
               align_to_words: bool = True):
        """
        Make predictions with the NER model.
        
        Args:
            encoded_inputs: Encoded inputs (from tokenizer)
            tokenizer: Tokenizer used to encode the inputs
            original_texts: Original texts corresponding to the inputs
            align_to_words: Whether to align predictions to words in original text
            
        Returns:
            Predictions as list of (entity, label) pairs for each text
        """
        self.model.eval()
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in encoded_inputs.items() 
                 if k != 'offset_mapping'}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
        
        batch_results = []
        
        # Process each sequence in the batch
        for i, pred in enumerate(predictions):
            sequence_pred = pred.cpu().numpy()
            
            if align_to_words and tokenizer and original_texts:
                # Get word-aligned entities (requires tokenizer and original text)
                # This is a complex process that depends on the tokenizer
                # Here we sketch the general approach:
                text = original_texts[i]
                
                # If we have offset mapping (from tokenizer with return_offsets_mapping=True)
                if 'offset_mapping' in encoded_inputs:
                    offset_mapping = encoded_inputs['offset_mapping'][i].numpy()
                    
                    # Group predictions by word using offsets
                    word_ids = []
                    current_word = None
                    for offsets in offset_mapping:
                        # Skip special tokens with no offsets
                        if offsets[0] == offsets[1] == 0:
                            word_ids.append(None)
                        elif current_word is not None and offsets[0] <= current_word[1]:
                            # Same word as previous token
                            word_ids.append(current_word[2])
                        else:
                            # New word
                            current_word = (offsets[0], offsets[1], len(word_ids))
                            word_ids.append(current_word[2])
                    
                    # Group predictions by word
                    word_predictions = []
                    prev_word_idx = None
                    for token_idx, word_idx in enumerate(word_ids):
                        if word_idx is not None:
                            # Skip special tokens
                            if prev_word_idx != word_idx:
                                # New word, take first token's prediction
                                word_predictions.append(sequence_pred[token_idx])
                            prev_word_idx = word_idx
                    
                    # Convert predictions to entity spans
                    entities = []
                    prev_label = 'O'
                    entity_start = 0
                    entity_label = ''
                    
                    for idx, label_id in enumerate(word_predictions):
                        label = self.id2label[label_id]
                        
                        # Handle B- and I- prefixes in BIO scheme
                        if label.startswith('B-'):
                            # End previous entity if there was one
                            if prev_label != 'O' and prev_label.startswith(('B-', 'I-')):
                                entities.append({
                                    'entity': text[entity_start:entity_end],
                                    'label': entity_label,
                                    'start': entity_start,
                                    'end': entity_end
                                })
                            
                            # Start new entity
                            entity_label = label[2:]  # Remove 'B-' prefix
                            entity_start = offset_mapping[idx][0]
                            entity_end = offset_mapping[idx][1]
                            
                        elif label.startswith('I-') and prev_label != 'O':
                            # Continue entity
                            entity_end = offset_mapping[idx][1]
                            
                        elif label == 'O':
                            # End previous entity if there was one
                            if prev_label != 'O' and prev_label.startswith(('B-', 'I-')):
                                entities.append({
                                    'entity': text[entity_start:entity_end],
                                    'label': entity_label,
                                    'start': entity_start,
                                    'end': entity_end
                                })
                        
                        prev_label = label
                    
                    # Add last entity if there is one
                    if prev_label != 'O' and prev_label.startswith(('B-', 'I-')):
                        entities.append({
                            'entity': text[entity_start:entity_end],
                            'label': entity_label,
                            'start': entity_start,
                            'end': entity_end
                        })
                    
                    batch_results.append(entities)
                else:
                    # Fallback if no offset mapping is available
                    batch_results.append(sequence_pred)
            else:
                # Return token-level predictions without alignment
                batch_results.append([self.id2label[p] for p in sequence_pred])
        
        return batch_results
    
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
            with open(os.path.join(path, 'label_mappings.json'), 'w') as f:
                json.dump({
                    'id2label': {str(k): v for k, v in self.id2label.items()},
                    'label2id': self.label2id
                }, f)
    
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
        
        label_mappings_path = os.path.join(path, 'label_mappings.json')
        if os.path.exists(label_mappings_path):
            import json
            with open(label_mappings_path, 'r') as f:
                mappings = json.load(f)
                id2label = {int(k): v for k, v in mappings['id2label'].items()}
                label2id = mappings['label2id']
        
        # Create instance
        instance = cls(
            model_name_or_path=path,
            num_labels=config.num_labels,
            id2label=id2label,
            label2id=label2id,
            device=device
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
    
    def __init__(self, 
                 model_name_or_path: str,
                 num_labels: int,
                 use_crf: bool = True,
                 use_char_features: bool = False,
                 char_vocab_size: Optional[int] = None,
                 char_embedding_dim: int = 30,
                 char_hidden_size: int = 50,
                 dropout: float = 0.1):
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
            num_labels=num_labels if not use_crf else None  # No classification head if using CRF
        )
        
        # Get hidden size from transformer
        hidden_size = self.transformer.config.hidden_size
        
        # Character-level features
        self.use_char_features = use_char_features
        if use_char_features:
            assert char_vocab_size is not None, "char_vocab_size must be specified for character features"
            
            self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
            self.char_lstm = nn.LSTM(
                char_embedding_dim,
                char_hidden_size,
                batch_first=True,
                bidirectional=True
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
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                char_ids=None, labels=None):
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
            token_type_ids=token_type_ids
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
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


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
    return NERModel(
        model_name_or_path=model_name,
        num_labels=num_labels,
        **kwargs
    )


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
        model_name_or_path=model_name,
        num_labels=num_labels,
        **kwargs
    )
