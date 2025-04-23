"""
Evaluation script for NLP models.

This script provides a command-line interface for evaluating NLP models
across different tasks using the enhanced architecture.
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
from src.models.classifier import TransformerClassifier
from src.models.named_entity_recognition import NERModel
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.summarizer import TextSummarizer, ExtractiveSummarizer
from src.training.metrics import compute_metrics_for_task, classification_report, token_classification_metrics, rouge_metrics
from src.utils.logging_utils import get_logger
from src.utils.visualization import plot_confusion_matrix, plot_classification_metrics, plot_token_classification_results

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate an NLP model")
    
    # Task and model selection
    parser.add_argument("--task", type=str, default="classification", 
                      choices=["classification", "ner", "sentiment", "summarization"],
                      help="NLP task")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model directory")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default=None, 
                      help="Dataset name from Hugging Face or path to local data")
    parser.add_argument("--data_dir", type=str, default=None,
                      help="Path to directory containing data files (if not using Hugging Face)")
    parser.add_argument("--split", type=str, default="test",
                      help="Dataset split to evaluate on")
    parser.add_argument("--text_column", type=str, default="text",
                      help="Column name containing text (for CSV/JSON data)")
    parser.add_argument("--label_column", type=str, default="label",
                      help="Column name containing labels (for CSV/JSON data)")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=16, 
                      help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum number of samples to evaluate")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="./results", 
                      help="Directory to save evaluation results")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations of evaluation results")
    parser.add_argument("--detailed_report", action="store_true",
                      help="Generate a detailed report with per-sample predictions")
    
    # Specific task parameters
    parser.add_argument("--summarization_metrics", action="store_true",
                      help="Calculate ROUGE metrics for summarization")
    
    return parser.parse_args()


def load_model(task, model_path):
    """Load a model for the specified task."""
    try:
        if task == "classification":
            model = TransformerClassifier.load(model_path)
        elif task == "ner":
            model = NERModel.load(model_path)
        elif task == "sentiment":
            model = SentimentAnalyzer.load(model_path)
        elif task == "summarization":
            # Check if it's an extractive model
            if os.path.exists(os.path.join(model_path, "extractive_method.txt")):
                with open(os.path.join(model_path, "extractive_method.txt"), "r") as f:
                    method = f.read().strip()
                model = ExtractiveSummarizer(method=method)
            else:
                model = TextSummarizer.load(model_path)
        else:
            logger.error(f"Unsupported task: {task}")
            return None
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return None


def evaluate_model(args):
    """Evaluate a model with the specified arguments."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.task, args.model_path)
    if model is None:
        return
    
    # Load tokenizer
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception:
        # Try to find tokenizer in a different location
        if os.path.exists(os.path.join(args.model_path, "tokenizer")):
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "tokenizer"))
        else:
            # For extractive summarization, we don't need a tokenizer
            tokenizer = None
            if args.task != "summarization" or not isinstance(model, ExtractiveSummarizer):
                logger.warning("Tokenizer not found. Some functionality may be limited.")
    
    # Create preprocessor
    preprocessor = TextPreprocessor()
    
    # Create data loader based on task
    if tokenizer:
        if args.task == "classification":
            dataset_loader = get_text_classification_loader(tokenizer, preprocessor)
        elif args.task == "ner":
            dataset_loader = get_ner_loader(tokenizer, preprocessor)
        elif args.task == "sentiment":
            dataset_loader = get_text_classification_loader(tokenizer, preprocessor)
        elif args.task == "summarization":
            dataset_loader = get_summarization_loader(tokenizer, preprocessor)
    else:
        # Basic dataset loader without tokenizer
        dataset_loader = NLPDatasetLoader(preprocessor=preprocessor, task_type=args.task)
    
    # Load dataset
    if args.dataset:
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
                label_column=args.label_column,
                split=args.split
            )
        
        # Preprocess dataset if tokenizer is available
        if tokenizer:
            dataset = dataset_loader.preprocess_dataset(dataset)
        
        # Limit samples if requested
        if args.max_samples and args.max_samples < len(dataset):
            import random
            random.seed(42)
            indices = random.sample(range(len(dataset)), args.max_samples)
            dataset = dataset.select(indices)
        
        # Create dataloader
        if tokenizer:
            dataloader = dataset_loader.create_torch_dataloaders(
                dataset,
                batch_size=args.batch_size
            )[args.split if args.split in dataset.keys() else 'train']
        
        # Evaluate model
        logger.info("Evaluating model...")
        
        if args.task == "classification" or args.task == "sentiment":
            # Use model's evaluate method if available
            if hasattr(model, 'evaluate') and tokenizer:
                eval_results = model.evaluate(dataloader)
                
                # Print results
                print("\n===== Evaluation Results =====")
                print(f"Loss: {eval_results['loss']:.4f}")
                print(f"Accuracy: {eval_results['accuracy']:.4f}")
                
                if 'per_class_metrics' in eval_results:
                    print("\n===== Per-Class Metrics =====")
                    for label, metrics in eval_results['per_class_metrics'].items():
                        if label != 'accuracy' and label != 'macro avg' and label != 'weighted avg':
                            print(f"{label}: Precision={metrics['precision']:.4f}, "
                                 f"Recall={metrics['recall']:.4f}, "
                                 f"F1={metrics['f1-score']:.4f}")
                
                # Save results
                results_path = os.path.join(args.output_dir, "evaluation_results.json")
                with open(results_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                
                # Generate visualizations if requested
                if args.visualize and tokenizer:
                    # Get predictions for visualization
                    import torch
                    import numpy as np
                    
                    all_preds = []
                    all_labels = []
                    all_probs = []
                    
                    model.model.eval()
                    for batch in dataloader:
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        with torch.no_grad():
                            outputs = model.model(**batch)
                        
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)
                        preds = torch.argmax(logits, dim=-1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch['labels'].cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                    
                    # Get class names if available
                    class_names = None
                    if hasattr(model, 'id2label'):
                        class_names = [model.id2label[i] for i in range(len(model.id2label))]
                    
                    # Plot confusion matrix
                    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
                    plot_confusion_matrix(
                        y_true=all_labels,
                        y_pred=all_preds,
                        class_names=class_names,
                        save_path=cm_path
                    )
                    
                    # Plot classification metrics
                    metrics_path = os.path.join(args.output_dir, 'classification_metrics.png')
                    plot_classification_metrics(
                        y_true=all_labels,
                        y_pred=all_preds,
                        y_proba=np.array(all_probs),
                        class_names=class_names,
                        save_path=metrics_path
                    )
                    
                    logger.info(f"Visualizations saved to {args.output_dir}")
            
            else:
                # Manual evaluation
                logger.warning("Model evaluate method not available. Using manual evaluation.")
                # Here we would implement manual evaluation
        
        elif args.task == "ner" and tokenizer:
            # Evaluate NER model
            if hasattr(model, 'evaluate'):
                eval_results = model.evaluate(dataloader)
                
                # Print results
                print("\n===== Evaluation Results =====")
                print(f"Loss: {eval_results['loss']:.4f}")
                print(f"F1 Score: {eval_results['f1']:.4f}")
                print(f"Precision: {eval_results['precision']:.4f}")
                print(f"Recall: {eval_results['recall']:.4f}")
                
                # Save results
                results_path = os.path.join(args.output_dir, "evaluation_results.json")
                with open(results_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                
                # Generate detailed report if requested
                if args.detailed_report and tokenizer:
                    # Get predictions for a few examples
                    examples = []
                    
                    # Get a few batches from the dataset
                    for batch_idx, batch in enumerate(dataloader):
                        if batch_idx >= 3:  # Limit to 3 batches
                            break
                        
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        
                        # Get original tokens
                        token_batch = [tokenizer.convert_ids_to_tokens(ids) for ids in batch['input_ids']]
                        
                        # Get predictions
                        outputs = model.model(**batch)
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        
                        # Extract token-level predictions and labels
                        for i, (tokens, preds, labels) in enumerate(zip(token_batch, predictions, batch['labels'])):
                            pred_list = []
                            label_list = []
                            token_list = []
                            
                            for j, (token, pred, label) in enumerate(zip(tokens, preds, labels)):
                                if label != -100:  # Skip padding tokens
                                    if hasattr(model, 'id2label'):
                                        pred_list.append(model.id2label[pred.item()])
                                        label_list.append(model.id2label[label.item()])
                                    else:
                                        pred_list.append(str(pred.item()))
                                        label_list.append(str(label.item()))
                                    
                                    token_list.append(token)
                            
                            # Add to examples
                            examples.append({
                                'tokens': token_list,
                                'true_labels': label_list,
                                'pred_labels': pred_list
                            })
                    
                    # Save examples
                    examples_path = os.path.join(args.output_dir, "ner_examples.json")
                    with open(examples_path, 'w') as f:
                        json.dump(examples, f, indent=2)
                    
                    # Visualize if requested
                    if args.visualize and examples:
                        for i, example in enumerate(examples):
                            if i >= 3:  # Limit to 3 visualizations
                                break
                            
                            viz_path = os.path.join(args.output_dir, f"ner_example_{i}.png")
                            plot_token_classification_results(
                                tokens=example['tokens'],
                                true_labels=example['true_labels'],
                                pred_labels=example['pred_labels'],
                                save_path=viz_path
                            )
                        
                        logger.info(f"NER visualizations saved to {args.output_dir}")
                
            else:
                logger.warning("Model evaluate method not available for NER.")
        
        elif args.task == "summarization":
            if args.summarization_metrics and tokenizer:
                # For abstractive summarization with ROUGE evaluation
                from src.training.metrics import rouge_metrics
                
                # Get texts and references from dataset
                texts = []
                references = []
                
                for item in dataset:
                    if 'text' in item and item['text']:
                        texts.append(item['text'])
                    if 'summary' in item and item['summary']:
                        references.append(item['summary'])
                    elif 'highlights' in item and item['highlights']:
                        references.append(item['highlights'])
                
                # Generate summaries
                if isinstance(model, TextSummarizer) and tokenizer:
                    # Abstractive summarization
                    summaries = model.summarize_text(
                        texts=texts[:min(len(texts), 50)],  # Limit to 50 texts
                        tokenizer=tokenizer,
                        max_length=128,
                        min_length=30
                    )
                elif isinstance(model, ExtractiveSummarizer):
                    # Extractive summarization
                    summaries = [model.summarize(text, ratio=0.3) for text in texts[:min(len(texts), 50)]]
                else:
                    logger.error("Unsupported summarization model type")
                    return
                
                # Calculate ROUGE scores
                if references and len(references) == len(summaries):
                    rouge_scores = rouge_metrics(summaries, references)
                    
                    # Print results
                    print("\n===== Summarization Results =====")
                    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
                    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
                    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
                    
                    # Save results
                    results_path = os.path.join(args.output_dir, "summarization_results.json")
                    with open(results_path, 'w') as f:
                        json.dump(rouge_scores, f, indent=2)
                    
                    # Save examples
                    if args.detailed_report:
                        examples = []
                        for i, (text, summary, reference) in enumerate(zip(texts, summaries, references)):
                            if i >= 10:  # Limit to 10 examples
                                break
                            
                            examples.append({
                                'text': text[:500] + "..." if len(text) > 500 else text,
                                'generated_summary': summary,
                                'reference_summary': reference
                            })
                        
                        examples_path = os.path.join(args.output_dir, "summarization_examples.json")
                        with open(examples_path, 'w') as f:
                            json.dump(examples, f, indent=2)
                else:
                    logger.warning("References not available for ROUGE calculation")
            else:
                logger.info("Summarization evaluation requires --summarization_metrics flag")
    else:
        logger.error("No dataset provided for evaluation")


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)