"""
Generic training utilities for NLP models.

This module provides base classes for model training across
different NLP tasks.
"""
import os
import time
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseTrainer:
    """
    Base trainer class for NLP models.
    
    This class provides a generic training loop that can be used
    across different NLP tasks.
    """
    
    def __init__(self, 
                 model,
                 train_dataloader: DataLoader,
                 eval_dataloader: Optional[DataLoader] = None,
                 optimizer: Optional[Optimizer] = None,
                 scheduler: Optional[_LRScheduler] = None,
                 device: Optional[str] = None,
                 max_grad_norm: float = 1.0,
                 evaluate_every: int = 100,
                 save_every: Optional[int] = None,
                 save_path: Optional[str] = None,
                 early_stopping: bool = False,
                 patience: int = 3,
                 metrics: Optional[List[str]] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            device: Device to use ('cpu', 'cuda', 'mps')
            max_grad_norm: Maximum gradient norm for gradient clipping
            evaluate_every: Number of steps between evaluations
            save_every: Number of steps between model saves
            save_path: Path to save the model
            early_stopping: Whether to use early stopping
            patience: Number of evaluations with no improvement before stopping
            metrics: List of metrics to compute during evaluation
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Training parameters
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=5e-5)
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.save_path = save_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.metrics = metrics or ['loss']
        
        # Training state variables
        self.global_step = 0
        self.best_eval_metric = float('inf')  # Lower is better (for loss)
        self.no_improvement_count = 0
        self.history = {metric: [] for metric in self.metrics}
        self.history['train_loss'] = []
        
        # Move model to device
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
    
    def train(self, num_epochs: int, **kwargs):
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            **kwargs: Additional arguments for customization
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            epoch_loss = self.train_epoch(epoch, **kwargs)
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s | "
                       f"Loss: {epoch_loss:.4f}")
            
            # Early stopping check
            if self.early_stopping and self.no_improvement_count >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model if path is provided
        if self.save_path and (self.save_every is None or self.global_step % self.save_every != 0):
            self.save_model(os.path.join(self.save_path, 'final_model'))
        
        return self.history
    
    def train_epoch(self, epoch: int, **kwargs):
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            **kwargs: Additional arguments for customization
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        
        for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update LR scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update training state
            epoch_loss += loss.item()
            self.history['train_loss'].append(loss.item())
            self.global_step += 1
            
            # Evaluate if needed
            if self.eval_dataloader is not None and self.global_step % self.evaluate_every == 0:
                eval_results = self.evaluate()
                
                # Print progress
                log_msg = f"Step {self.global_step} | Train Loss: {loss.item():.4f}"
                for metric, value in eval_results.items():
                    log_msg += f" | {metric}: {value:.4f}"
                logger.info(log_msg)
                
                # Early stopping check
                main_metric = self.metrics[0]  # First metric is used for early stopping
                current_metric = eval_results.get(main_metric, float('inf'))
                
                if current_metric < self.best_eval_metric:
                    self.best_eval_metric = current_metric
                    self.no_improvement_count = 0
                    
                    # Save best model
                    if self.save_path:
                        self.save_model(os.path.join(self.save_path, 'best_model'))
                else:
                    self.no_improvement_count += 1
                
                # Return to training mode
                self.model.train()
            
            # Save model if needed
            if self.save_path and self.save_every and self.global_step % self.save_every == 0:
                self.save_model(os.path.join(self.save_path, f'model_step_{self.global_step}'))
        
        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        return avg_epoch_loss
    
    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        eval_loss = 0
        
        # Store predictions and labels for metric calculation
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()
                
                # Store predictions and labels if logits are available
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    
                    if logits.size(-1) > 1:  # Classification
                        preds = torch.argmax(logits, dim=-1)
                    else:  # Regression
                        preds = logits.squeeze(-1)
                    
                    all_preds.append(preds.cpu())
                    
                    if 'labels' in batch:
                        all_labels.append(batch['labels'].cpu())
        
        # Calculate metrics
        metrics = {'loss': eval_loss / len(self.eval_dataloader)}
        
        # Concatenate predictions and labels
        if all_preds and all_labels:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Calculate additional metrics based on task
            metrics.update(self.calculate_metrics(all_preds, all_labels))
        
        # Store metrics in history
        for metric, value in metrics.items():
            if metric in self.history:
                self.history[metric].append(value)
        
        return metrics
    
    def calculate_metrics(self, preds, labels):
        """
        Calculate evaluation metrics.
        
        Args:
            preds: Model predictions
            labels: Ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        # This method should be implemented by task-specific subclasses
        # Default implementation: accuracy for classification
        metrics = {}
        
        if 'accuracy' in self.metrics:
            accuracy = (preds == labels).float().mean().item()
            metrics['accuracy'] = accuracy
        
        return metrics
    
    def save_model(self, path: str):
        """
        Save the model and training state.
        
        Args:
            path: Path to save the model
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            # Fallback for models without save_pretrained
            torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'best_eval_metric': self.best_eval_metric,
            'no_improvement_count': self.no_improvement_count,
            'optimizer': self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            training_state['scheduler'] = self.scheduler.state_dict()
        
        torch.save(training_state, os.path.join(path, 'training_state.pt'))
        
        # Save training history
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            # Convert numpy values to float for JSON serialization
            json_safe_history = {}
            for k, v in self.history.items():
                json_safe_history[k] = [float(x) if isinstance(x, (np.number, np.ndarray)) else x for x in v]
            json.dump(json_safe_history, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a model and training state.
        
        Args:
            path: Path to load the model from
        """
        # Load model
        if hasattr(self.model, 'from_pretrained'):
            self.model = self.model.from_pretrained(path)
            self.model.to(self.device)
        else:
            # Fallback for models without from_pretrained
            self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        
        # Load training state
        if os.path.exists(os.path.join(path, 'training_state.pt')):
            training_state = torch.load(os.path.join(path, 'training_state.pt'))
            
            self.global_step = training_state['global_step']
            self.best_eval_metric = training_state['best_eval_metric']
            self.no_improvement_count = training_state['no_improvement_count']
            
            self.optimizer.load_state_dict(training_state['optimizer'])
            
            if self.scheduler is not None and 'scheduler' in training_state:
                self.scheduler.load_state_dict(training_state['scheduler'])
        
        # Load training history
        if os.path.exists(os.path.join(path, 'training_history.json')):
            with open(os.path.join(path, 'training_history.json'), 'r') as f:
                self.history = json.load(f)
        
        logger.info(f"Model loaded from {path}")


class ClassificationTrainer(BaseTrainer):
    """
    Trainer for text classification tasks.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the classification trainer."""
        # Set default metrics for classification
        kwargs['metrics'] = kwargs.get('metrics', ['loss', 'accuracy', 'f1'])
        super().__init__(*args, **kwargs)
    
    def calculate_metrics(self, preds, labels):
        """Calculate classification-specific metrics."""
        metrics = {}
        
        # Accuracy
        if 'accuracy' in self.metrics:
            accuracy = (preds == labels).float().mean().item()
            metrics['accuracy'] = accuracy
        
        # F1 Score
        if 'f1' in self.metrics:
            try:
                from sklearn.metrics import f1_score
                
                # Convert to numpy for sklearn
                preds_np = preds.numpy()
                labels_np = labels.numpy()
                
                # Check if binary or multi-class
                if len(set(labels_np)) <= 2:
                    f1 = f1_score(labels_np, preds_np)
                else:
                    f1 = f1_score(labels_np, preds_np, average='weighted')
                
                metrics['f1'] = f1
            except ImportError:
                logger.warning("scikit-learn not found. F1 score not computed.")
        
        # Precision and Recall
        if 'precision' in self.metrics or 'recall' in self.metrics:
            try:
                from sklearn.metrics import precision_score, recall_score
                
                # Convert to numpy for sklearn
                preds_np = preds.numpy()
                labels_np = labels.numpy()
                
                # Check if binary or multi-class
                if len(set(labels_np)) <= 2:
                    precision = precision_score(labels_np, preds_np)
                    recall = recall_score(labels_np, preds_np)
                else:
                    precision = precision_score(labels_np, preds_np, average='weighted')
                    recall = recall_score(labels_np, preds_np, average='weighted')
                
                metrics['precision'] = precision
                metrics['recall'] = recall
            except ImportError:
                logger.warning("scikit-learn not found. Precision/Recall not computed.")
        
        return metrics


class NERTrainer(BaseTrainer):
    """
    Trainer for named entity recognition tasks.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the NER trainer."""
        # Set default metrics for NER
        kwargs['metrics'] = kwargs.get('metrics', ['loss', 'f1', 'precision', 'recall'])
        super().__init__(*args, **kwargs)
        
        # Token mapping for handling padded tokens
        self.ignore_index = -100
    
    def evaluate(self):
        """Evaluate the NER model."""
        self.model.eval()
        eval_loss = 0
        
        # For token classification, we need to handle the special case
        true_predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # Extract token-level predictions and labels, ignoring padding
                for i, label in enumerate(batch["labels"]):
                    pred_list = []
                    label_list = []
                    
                    for j, lbl in enumerate(label):
                        if lbl != self.ignore_index:
                            pred_list.append(predictions[i, j].item())
                            label_list.append(lbl.item())
                    
                    true_predictions.append(pred_list)
                    true_labels.append(label_list)
        
        # Calculate metrics
        metrics = {'loss': eval_loss / len(self.eval_dataloader)}
        
        # Calculate additional NER metrics
        metrics.update(self.calculate_ner_metrics(true_predictions, true_labels))
        
        # Store metrics in history
        for metric, value in metrics.items():
            if metric in self.history:
                self.history[metric].append(value)
        
        return metrics
    
    def calculate_ner_metrics(self, predictions, labels):
        """
        Calculate NER-specific metrics.
        
        Args:
            predictions: List of token-level predictions
            labels: List of token-level labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert numeric labels to strings if id2label is available
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
            id2label = self.model.config.id2label
            str_predictions = [
                [id2label[p] for p in pred]
                for pred in predictions
            ]
            str_labels = [
                [id2label[l] for l in label]
                for label in labels
            ]
        else:
            # Use numeric labels as is
            str_predictions = predictions
            str_labels = labels
        
        # Calculate metrics using seqeval
        try:
            from seqeval.metrics import f1_score, precision_score, recall_score
            
            if 'f1' in self.metrics:
                metrics['f1'] = f1_score(str_labels, str_predictions)
            
            if 'precision' in self.metrics:
                metrics['precision'] = precision_score(str_labels, str_predictions)
            
            if 'recall' in self.metrics:
                metrics['recall'] = recall_score(str_labels, str_predictions)
                
        except ImportError:
            logger.warning("seqeval not found. NER metrics not computed.")
        
        return metrics


class SummarizationTrainer(BaseTrainer):
    """
    Trainer for text summarization tasks.
    """
    
    def __init__(self, *args, tokenizer=None, compute_rouge=True, **kwargs):
        """
        Initialize the summarization trainer.
        
        Args:
            *args: Arguments for BaseTrainer
            tokenizer: Tokenizer for decoding predictions
            compute_rouge: Whether to compute ROUGE scores
            **kwargs: Additional arguments for BaseTrainer
        """
        # Set default metrics for summarization
        kwargs['metrics'] = kwargs.get('metrics', ['loss', 'rouge1', 'rouge2', 'rougeL'])
        super().__init__(*args, **kwargs)
        
        self.tokenizer = tokenizer
        self.compute_rouge = compute_rouge
    
    def evaluate(self):
        """Evaluate the summarization model."""
        self.model.eval()
        eval_loss = 0
        
        # For seq2seq tasks, we need to generate summaries and compute ROUGE
        decoded_preds = []
        decoded_labels = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()
                
                # Generate summaries if needed for ROUGE
                if self.compute_rouge and self.tokenizer:
                    # Generate predictions
                    generated_ids = self.model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    # Decode predictions and labels
                    preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                    
                    decoded_preds.extend(preds)
                    decoded_labels.extend(labels)
        
        # Calculate metrics
        metrics = {'loss': eval_loss / len(self.eval_dataloader)}
        
        # Calculate ROUGE scores if needed
        if self.compute_rouge and self.tokenizer and decoded_preds:
            metrics.update(self.calculate_rouge_metrics(decoded_preds, decoded_labels))
        
        # Store metrics in history
        for metric, value in metrics.items():
            if metric in self.history:
                self.history[metric].append(value)
        
        return metrics
    
    def calculate_rouge_metrics(self, predictions, references):
        """
        Calculate ROUGE metrics for summarization.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
            
        Returns:
            Dictionary of ROUGE metrics
        """
        metrics = {}
        
        try:
            from rouge_score import rouge_scorer
            
            # Initialize ROUGE scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # Calculate ROUGE scores
            rouge_scores = {
                'rouge1': [],
                'rouge2': [],
                'rougeL': []
            }
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                rouge_scores['rouge1'].append(score['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(score['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(score['rougeL'].fmeasure)
            
            # Average scores
            metrics['rouge1'] = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
            metrics['rouge2'] = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
            metrics['rougeL'] = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
            
        except ImportError:
            logger.warning("rouge_score package not found. ROUGE metrics not computed.")
        
        return metrics


# Factory functions
def create_trainer(task: str, model, **kwargs):
    """
    Create a trainer for a specific task.
    
    Args:
        task: NLP task ('classification', 'ner', 'summarization', etc.)
        model: Model to train
        **kwargs: Additional arguments for the trainer
        
    Returns:
        Task-specific trainer instance
    """
    if task.lower() == 'classification':
        return ClassificationTrainer(model=model, **kwargs)
    elif task.lower() in ('ner', 'token_classification'):
        return NERTrainer(model=model, **kwargs)
    elif task.lower() in ('summarization', 'translation'):
        return SummarizationTrainer(model=model, **kwargs)
    else:
        logger.warning(f"Unknown task: {task}. Using BaseTrainer.")
        return BaseTrainer(model=model, **kwargs)
