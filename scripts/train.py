"""
Training script for NLP models.

This script provides a command-line interface for training various NLP models
using the enhanced architecture.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import NLPDatasetLoader, get_text_classification_loader, get_ner_loader, get_summarization_loader
from src.data.preprocessing import TextPreprocessor
from src.models.classifier import create_classifier, create_ensemble
from src.models.named_entity_recognition import create_ner_model
from src.models.sentiment_analyzer import create_sentiment_analyzer
from src.models.summarizer import create_abstractive_summarizer, create_extractive_summarizer
from src.training.trainer import create_trainer
from src.utils.logging_utils import get_logger, ExperimentTracker
from src.utils.visualization import plot_training_history, plot_confusion_matrix

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an NLP model")
    
    # Task and model selection
    parser.add_argument("--task", type=str, default="classification", 
                      choices=["classification", "ner", "sentiment", "summarization"],
                      help="NLP task")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", 
                      help="Model name from Hugging Face or model configuration")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="imdb", 
                      help="Dataset name from Hugging Face or path to local data")
    parser.add_argument("--data_dir", type=str, default=None,
                      help="Path to directory containing data files (if not using Hugging Face)")
    parser.add_argument("--text_column", type=str, default="text",
                      help="Column name containing text (for CSV/JSON data)")
    parser.add_argument("--label_column", type=str, default="label",
                      help="Column name containing labels (for CSV/JSON data)")
    
    # Training parameters
    parser.add_argument("--config_file", type=str, default=None,
                      help="Path to model configuration file")
    parser.add_argument("--epochs", type=int, default=3, 
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                      help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, 
                      help="Maximum sequence length")
    parser.add_argument("--num_labels", type=int, default=None,
                      help="Number of labels/classes")
    
    # Preprocessing options
    parser.add_argument("--lowercase", action="store_true", 
                      help="Convert text to lowercase")
    parser.add_argument("--remove_punctuation", action="store_true", 
                      help="Remove punctuation")
    parser.add_argument("--remove_stopwords", action="store_true", 
                      help="Remove stopwords")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="./models", 
                      help="Directory to save the model")
    parser.add_argument("--experiment_name", type=str, default=None,
                      help="Name for the experiment")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations of training")
    
    # Advanced options
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                      help="Use mixed precision training")
    parser.add_argument("--early_stopping", action="store_true",
                      help="Use early stopping")
    parser.add_argument("--patience", type=int, default=3,
                      help="Patience for early stopping")
    
    return parser.parse_args()


def load_config(config_file):
    """Load configuration from a JSON file."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


def train_model(args):
    """Train a model with the specified arguments."""
    # Set up experiment tracking
    experiment_name = args.experiment_name or f"{args.task}_{args.model_name.split('/')[-1]}"
    experiment = ExperimentTracker(
        experiment_name=experiment_name,
        base_dir=args.output_dir
    )
    
    # Log parameters
    params = vars(args)
    experiment.log_parameters(params)
    
    # Load configuration if provided
    config = load_config(args.config_file)
    
    # Determine number of labels if not specified
    num_labels = args.num_labels
    if num_labels is None:
        if args.task == "classification" or args.task == "sentiment":
            # Default to binary classification if not specified
            num_labels = 2
        elif args.task == "ner":
            # Default to CoNLL-2003 NER (9 entity types including 'O')
            num_labels = 9
    
    # Create preprocessor
    preprocessor = TextPreprocessor(
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        remove_stopwords=args.remove_stopwords
    )
    
    # Load tokenizer based on model name
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return
    
    # Create data loader based on task
    if args.task == "classification":
        dataset_loader = get_text_classification_loader(tokenizer, preprocessor)
    elif args.task == "ner":
        dataset_loader = get_ner_loader(tokenizer, preprocessor)
    elif args.task == "sentiment":
        dataset_loader = get_text_classification_loader(tokenizer, preprocessor)
    elif args.task == "summarization":
        dataset_loader = get_summarization_loader(tokenizer, preprocessor)
    else:
        logger.error(f"Unsupported task: {args.task}")
        return
    
    # Load dataset
    if os.path.isfile(args.dataset):
        # Load from file
        if args.dataset.endswith('.csv'):
            dataset = dataset_loader.load_from_csv(
                args.dataset,
                text_column=args.text_column,
                label_column=args.label_column
            )
        elif args.dataset.endswith('.json'):
            dataset = dataset_loader.load_from_json(
                args.dataset,
                text_key=args.text_column,
                label_key=args.label_column
            )
        else:
            logger.error(f"Unsupported file format: {args.dataset}")
            return
    else:
        # Load from Hugging Face datasets
        dataset = dataset_loader.load_huggingface_dataset(
            args.dataset,
            text_column=args.text_column,
            label_column=args.label_column
        )
    
    # Preprocess dataset
    dataset = dataset_loader.preprocess_dataset(dataset)
    
    # Create data loaders
    dataloaders = dataset_loader.create_torch_dataloaders(
        dataset,
        batch_size=args.batch_size
    )
    
    # Create model based on task
    if args.task == "classification":
        model = create_classifier(
            model_name=args.model_name,
            num_labels=num_labels
        )
    elif args.task == "ner":
        model = create_ner_model(
            model_name=args.model_name,
            num_labels=num_labels
        )
    elif args.task == "sentiment":
        model = create_sentiment_analyzer(
            model_name=args.model_name,
            num_labels=num_labels
        )
    elif args.task == "summarization":
        model = create_abstractive_summarizer(
            model_name=args.model_name
        )
    
    # Create optimizer
    from torch.optim import AdamW
    optimizer = AdamW(
        model.model.parameters(),
        lr=args.learning_rate
    )
    
    # Create trainer
    trainer = create_trainer(
        task=args.task,
        model=model.model,
        train_dataloader=dataloaders['train'],
        eval_dataloader=dataloaders.get('validation', None),
        optimizer=optimizer,
        max_grad_norm=1.0,
        early_stopping=args.early_stopping,
        patience=args.patience,
        save_path=experiment.model_dir
    )
    
    # Train model
    history = trainer.train(args.epochs)
    
    # Log final metrics
    if 'validation' in dataloaders:
        eval_results = model.evaluate(dataloaders['validation'])
        experiment.log_metrics(eval_results, args.epochs, prefix='final_')
    
    # Save model
    model_path = os.path.join(experiment.model_dir, 'final_model')
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        # Plot training history
        history_plot_path = os.path.join(experiment.artifacts_dir, 'training_history.png')
        plot_training_history(history, save_path=history_plot_path)
        
        # Plot confusion matrix if applicable
        if args.task in ("classification", "sentiment") and 'validation' in dataloaders:
            import torch
            import numpy as np
            
            # Get predictions
            all_preds = []
            all_labels = []
            
            model.model.eval()
            for batch in dataloaders['validation']:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model.model(**batch)
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
            
            # Plot confusion matrix
            cm_path = os.path.join(experiment.artifacts_dir, 'confusion_matrix.png')
            
            # Get class names if available
            class_names = None
            if hasattr(model, 'id2label'):
                class_names = [model.id2label[i] for i in range(len(model.id2label))]
            
            plot_confusion_matrix(
                y_true=all_labels,
                y_pred=all_preds,
                class_names=class_names,
                save_path=cm_path
            )
    
    return model, tokenizer, experiment


if __name__ == "__main__":
    args = parse_args()
    train_model(args)