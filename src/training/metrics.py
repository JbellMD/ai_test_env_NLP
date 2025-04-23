"""
Evaluation metrics for NLP tasks.

This module provides functions for calculating various evaluation metrics
for different NLP tasks such as classification, named entity recognition,
text summarization, etc.
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Import optional dependencies if available
try:
    import sklearn.metrics as sk_metrics
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from seqeval.metrics import f1_score as seqeval_f1_score
    from seqeval.metrics import precision_score as seqeval_precision_score
    from seqeval.metrics import recall_score as seqeval_recall_score
    from seqeval.metrics import classification_report as seqeval_report
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# General metrics
def accuracy(preds, labels):
    """
    Calculate accuracy.
    
    Args:
        preds: Predictions
        labels: Ground truth labels
        
    Returns:
        Accuracy score
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    return (preds == labels).mean()


def f1_score(preds, labels, average='weighted'):
    """
    Calculate F1 score.
    
    Args:
        preds: Predictions
        labels: Ground truth labels
        average: Averaging method
        
    Returns:
        F1 score
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for F1 score calculation")
    
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    return sk_metrics.f1_score(labels, preds, average=average)


def precision_score(preds, labels, average='weighted'):
    """
    Calculate precision.
    
    Args:
        preds: Predictions
        labels: Ground truth labels
        average: Averaging method
        
    Returns:
        Precision score
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for precision calculation")
    
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    return sk_metrics.precision_score(labels, preds, average=average)


def recall_score(preds, labels, average='weighted'):
    """
    Calculate recall.
    
    Args:
        preds: Predictions
        labels: Ground truth labels
        average: Averaging method
        
    Returns:
        Recall score
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for recall calculation")
    
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    return sk_metrics.recall_score(labels, preds, average=average)


def classification_report(preds, labels, target_names=None):
    """
    Generate a classification report.
    
    Args:
        preds: Predictions
        labels: Ground truth labels
        target_names: List of class names
        
    Returns:
        Classification report
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for classification report")
    
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    return sk_metrics.classification_report(labels, preds, target_names=target_names, output_dict=True)


# NER-specific metrics
def token_classification_metrics(preds, labels, id2label=None):
    """
    Calculate token classification metrics.
    
    Args:
        preds: Predictions (token IDs or labels)
        labels: Ground truth labels (token IDs or labels)
        id2label: Mapping from IDs to label names
        
    Returns:
        Dictionary of metrics
    """
    if not SEQEVAL_AVAILABLE:
        raise ImportError("seqeval is required for token classification metrics")
    
    # Convert IDs to labels if needed
    if id2label is not None:
        if isinstance(preds[0][0], (int, np.integer)):
            str_preds = [[id2label[p] for p in pred] for pred in preds]
        else:
            str_preds = preds
            
        if isinstance(labels[0][0], (int, np.integer)):
            str_labels = [[id2label[l] for l in label] for label in labels]
        else:
            str_labels = labels
    else:
        str_preds = preds
        str_labels = labels
    
    # Calculate metrics
    metrics = {
        'f1': seqeval_f1_score(str_labels, str_preds),
        'precision': seqeval_precision_score(str_labels, str_preds),
        'recall': seqeval_recall_score(str_labels, str_preds)
    }
    
    # Detailed report
    metrics['report'] = seqeval_report(str_labels, str_preds, output_dict=True)
    
    return metrics


# Summarization-specific metrics
def rouge_metrics(predictions, references, rouge_types=None):
    """
    Calculate ROUGE metrics for text summarization.
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
        rouge_types: ROUGE types to calculate
        
    Returns:
        Dictionary of ROUGE metrics
    """
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge_score is required for ROUGE metrics")
    
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    # Calculate scores for each pair
    scores = {rouge_type: [] for rouge_type in rouge_types}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        
        for rouge_type in rouge_types:
            scores[rouge_type].append(score[rouge_type].fmeasure)
    
    # Calculate averages
    results = {
        rouge_type: sum(scores[rouge_type]) / len(scores[rouge_type])
        for rouge_type in rouge_types
    }
    
    return results


# Translation-specific metrics
def bleu_score(predictions, references, max_order=4, smooth=False):
    """
    Calculate BLEU score for machine translation.
    
    Args:
        predictions: Generated translations
        references: Reference translations (can be multiple per example)
        max_order: Maximum n-gram order to use
        smooth: Whether to apply smoothing
        
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        raise ImportError("nltk is required for BLEU score calculation")
    
    # Tokenize if needed
    if isinstance(predictions[0], str):
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        predictions = [nltk.word_tokenize(pred.lower()) for pred in predictions]
    
    if isinstance(references[0][0], str):  # references is list of list of strings
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        references = [[nltk.word_tokenize(ref.lower()) for ref in refs] for refs in references]
    
    # Prepare for corpus_bleu
    if smooth:
        smoothing_function = SmoothingFunction().method1
    else:
        smoothing_function = None
    
    # Calculate BLEU
    weights = [1.0 / max_order] * max_order
    
    return corpus_bleu(
        references,
        predictions,
        weights=weights,
        smoothing_function=smoothing_function
    )


# Metric aggregation functions
def compute_metrics_for_task(task, preds, labels, **kwargs):
    """
    Compute metrics for a specific task.
    
    Args:
        task: NLP task name
        preds: Predictions
        labels: Ground truth labels
        **kwargs: Additional arguments for specific metrics
        
    Returns:
        Dictionary of metrics
    """
    if task.lower() == 'classification':
        metrics = {
            'accuracy': accuracy(preds, labels)
        }
        
        if SKLEARN_AVAILABLE:
            metrics.update({
                'f1': f1_score(preds, labels),
                'precision': precision_score(preds, labels),
                'recall': recall_score(preds, labels)
            })
            
            if 'target_names' in kwargs:
                metrics['report'] = classification_report(preds, labels, target_names=kwargs['target_names'])
    
    elif task.lower() in ('ner', 'token_classification'):
        if SEQEVAL_AVAILABLE:
            metrics = token_classification_metrics(preds, labels, id2label=kwargs.get('id2label'))
        else:
            metrics = {'accuracy': accuracy(preds, labels)} if isinstance(preds[0], (int, np.integer)) else {}
    
    elif task.lower() == 'summarization':
        if ROUGE_AVAILABLE and isinstance(preds[0], str):
            metrics = rouge_metrics(preds, labels, rouge_types=kwargs.get('rouge_types'))
        else:
            metrics = {'accuracy': accuracy(preds, labels)} if isinstance(preds[0], (int, np.integer)) else {}
    
    elif task.lower() == 'translation':
        if 'references' in kwargs:  # Multiple references per example
            references = kwargs['references']
            metrics = {'bleu': bleu_score(preds, references)}
        else:
            # Format single references as list of single-element lists
            refs_for_bleu = [[ref] for ref in labels]
            metrics = {'bleu': bleu_score(preds, refs_for_bleu)}
    
    else:
        # Default: just use accuracy if possible
        if isinstance(preds[0], (int, np.integer)) and isinstance(labels[0], (int, np.integer)):
            metrics = {'accuracy': accuracy(preds, labels)}
        else:
            metrics = {}
    
    return metrics


# Custom metrics for specific scenarios
def perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()


def exact_match(preds, labels):
    """
    Calculate exact match score (for QA tasks).
    
    Args:
        preds: Predicted answers
        labels: Ground truth answers
        
    Returns:
        Exact match score
    """
    # Normalize before comparison
    def normalize(text):
        return text.strip().lower()
    
    if isinstance(preds[0], str):
        normalized_preds = [normalize(pred) for pred in preds]
        normalized_labels = [normalize(label) for label in labels]
        
        return sum(p == l for p, l in zip(normalized_preds, normalized_labels)) / len(preds)
    else:
        return accuracy(preds, labels)


def f1_token_level(preds, labels):
    """
    Calculate token-level F1 score (for QA tasks).
    
    Args:
        preds: Predicted answers
        labels: Ground truth answers
        
    Returns:
        Token-level F1 score
    """
    if not isinstance(preds[0], str):
        return f1_score(preds, labels)
    
    f1_scores = []
    
    for pred, label in zip(preds, labels):
        # Tokenize
        pred_tokens = set(pred.lower().split())
        label_tokens = set(label.lower().split())
        
        # Calculate precision, recall, F1
        if not pred_tokens and not label_tokens:
            f1_scores.append(1.0)  # Both empty
        elif not pred_tokens or not label_tokens:
            f1_scores.append(0.0)  # One is empty
        else:
            common_tokens = pred_tokens.intersection(label_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(label_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
    
    return sum(f1_scores) / len(f1_scores)


# Metrics registry
METRICS_REGISTRY = {
    'accuracy': accuracy,
    'f1': f1_score,
    'precision': precision_score,
    'recall': recall_score,
    'classification_report': classification_report,
    'token_classification_metrics': token_classification_metrics,
    'rouge': rouge_metrics,
    'bleu': bleu_score,
    'perplexity': perplexity,
    'exact_match': exact_match,
    'f1_token_level': f1_token_level
}


def get_metric_fn(metric_name):
    """
    Get a metric function by name.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Metric function
    """
    if metric_name not in METRICS_REGISTRY:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    return METRICS_REGISTRY[metric_name]
