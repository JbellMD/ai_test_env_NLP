"""
Visualization utilities for NLP tasks.

This module provides functions for visualizing model outputs,
attention weights, training metrics, and other NLP-related data.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from .logging_utils import get_logger

logger = get_logger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Plot training history metrics.

    Args:
        history: Dictionary of metrics (keys) and their values over training
        figsize: Figure size
        save_path: Path to save the plot (if None, the plot is displayed)
    """
    plt.figure(figsize=figsize)

    # Determine how many subplots we need
    n_metrics = len(history.keys())
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each metric
    for i, (metric_name, values) in enumerate(history.items()):
        if i < len(axes):
            ax = axes[i]
            ax.plot(values)
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.7)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved training history plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    normalize: bool = False,
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix for classification results.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        figsize: Figure size
        cmap: Colormap for the plot
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot
    """
    # Convert to numpy arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        cbar=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention(
    attention_weights: np.ndarray,
    tokens: List[str],
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """
    Plot attention weights.

    Args:
        attention_weights: Attention weight matrix (n_tokens x n_tokens)
        tokens: Token strings
        figsize: Figure size
        cmap: Colormap for the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)

    # Plot attention weights
    sns.heatmap(
        attention_weights,
        annot=False,
        cmap=cmap,
        xticklabels=tokens,
        yticklabels=tokens,
        square=True,
    )

    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.title("Attention Weights")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved attention weights plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_token_classification_results(
    tokens: List[str],
    true_labels: Optional[List[str]] = None,
    pred_labels: List[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
):
    """
    Visualize token classification results (e.g., for NER).

    Args:
        tokens: List of tokens
        true_labels: List of true labels (optional)
        pred_labels: List of predicted labels
        figsize: Figure size
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle missing labels
    if true_labels is None:
        true_labels = ["O"] * len(tokens)
    if pred_labels is None:
        pred_labels = ["O"] * len(tokens)

    # Determine all unique labels
    all_labels = sorted(set(true_labels + pred_labels))

    # Create color map for labels
    cmap = plt.cm.get_cmap("tab20", len(all_labels))
    colors = {label: cmap(i) for i, label in enumerate(all_labels)}

    # Plot tokens and color-code based on labels
    for i, (token, true_label, pred_label) in enumerate(
        zip(tokens, true_labels, pred_labels)
    ):
        # Position for token
        pos = (i, 0.6)

        # Plot token text
        ax.text(pos[0], pos[1], token, ha="center", va="center", fontsize=12)

        # Plot true label above (if available)
        if true_labels is not None and true_label != "O":
            ax.text(
                pos[0],
                pos[1] + 0.2,
                true_label,
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                bbox=dict(facecolor=colors[true_label], alpha=0.5),
            )

        # Plot predicted label below
        if pred_label != "O":
            ax.text(
                pos[0],
                pos[1] - 0.2,
                pred_label,
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                bbox=dict(facecolor=colors[pred_label], alpha=0.5),
            )

    # Add legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in all_labels]
    ax.legend(
        handles,
        all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=min(6, len(all_labels)),
    )

    # Configure axis
    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(0, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Token Classification Results")

    # Add labels for true and predicted
    ax.text(-0.5, 0.8, "True:", ha="right", va="center", fontsize=12, fontweight="bold")
    ax.text(-0.5, 0.4, "Pred:", ha="right", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved token classification results plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
):
    """
    Plot various classification metrics (PR curve, ROC curve, confusion matrix).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for PR and ROC curves)
        class_names: Names of the classes
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Convert to numpy arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix")

    # Handle PR and ROC curves if probabilities are provided
    if y_proba is not None:
        # For binary classification
        if y_proba.shape[1] == 2:
            # 2. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            axes[1].plot(recall, precision, "b-", linewidth=2)
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Curve")
            axes[1].grid(True, linestyle="--", alpha=0.7)

            # 3. ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            axes[2].plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {roc_auc:.2f}")
            axes[2].plot([0, 1], [0, 1], "k--", alpha=0.7)
            axes[2].set_xlabel("False Positive Rate")
            axes[2].set_ylabel("True Positive Rate")
            axes[2].set_title("ROC Curve")
            axes[2].grid(True, linestyle="--", alpha=0.7)
            axes[2].legend()

        # For multi-class classification
        else:
            n_classes = y_proba.shape[1]

            # 2. Precision-Recall Curve (one vs rest)
            for i in range(n_classes):
                y_true_binary = (y_true == i).astype(int)
                precision, recall, _ = precision_recall_curve(
                    y_true_binary, y_proba[:, i]
                )
                class_name = class_names[i] if class_names else f"Class {i}"
                axes[1].plot(recall, precision, linewidth=2, label=class_name)

            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Curves (One vs Rest)")
            axes[1].grid(True, linestyle="--", alpha=0.7)
            axes[1].legend()

            # 3. ROC Curve (one vs rest)
            for i in range(n_classes):
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                class_name = class_names[i] if class_names else f"Class {i}"
                axes[2].plot(
                    fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {roc_auc:.2f})"
                )

            axes[2].plot([0, 1], [0, 1], "k--", alpha=0.7)
            axes[2].set_xlabel("False Positive Rate")
            axes[2].set_ylabel("True Positive Rate")
            axes[2].set_title("ROC Curves (One vs Rest)")
            axes[2].grid(True, linestyle="--", alpha=0.7)
            axes[2].legend()

    else:
        # If no probabilities, show empty plots for PR and ROC
        for i in range(1, 3):
            axes[i].text(
                0.5,
                0.5,
                "Probabilities required\nfor this plot",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[i].set_xticks([])
            axes[i].set_yticks([])

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved classification metrics plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_summarization_results(
    texts: List[str],
    summaries: List[str],
    reference_summaries: Optional[List[str]] = None,
    rouge_scores: Optional[List[Dict[str, float]]] = None,
    figsize: Tuple[int, int] = (15, 10),
    max_chars: int = 300,
    save_path: Optional[str] = None,
):
    """
    Visualize summarization results.

    Args:
        texts: List of original texts
        summaries: List of generated summaries
        reference_summaries: List of reference summaries
        rouge_scores: ROUGE scores for each summary
        figsize: Figure size
        max_chars: Maximum characters to display
        save_path: Path to save the plot
    """
    # Determine how many examples to show
    n_examples = min(len(texts), 4)  # Show at most 4 examples

    # Create figure
    fig, axes = plt.subplots(n_examples, 1, figsize=figsize)
    if n_examples == 1:
        axes = [axes]

    for i in range(n_examples):
        # Prepare text display (truncate if needed)
        text = texts[i]
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        summary = summaries[i]
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."

        # Prepare display text
        display_text = f"Text: {text}\n\nSummary: {summary}"

        if reference_summaries is not None:
            ref_summary = reference_summaries[i]
            if len(ref_summary) > max_chars:
                ref_summary = ref_summary[:max_chars] + "..."
            display_text += f"\n\nReference: {ref_summary}"

        if rouge_scores is not None:
            rouge = rouge_scores[i]
            rouge_text = ", ".join(f"{k}: {v:.4f}" for k, v in rouge.items())
            display_text += f"\n\nROUGE Scores: {rouge_text}"

        # Display text in the subplot
        axes[i].text(
            0.05, 0.95, display_text, va="top", ha="left", fontsize=10, wrap=True
        )
        axes[i].axis("off")

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved summarization results plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_word_importances(
    text: str,
    word_importances: Union[List[float], np.ndarray],
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = "coolwarm",
    save_path: Optional[str] = None,
):
    """
    Visualize word importances (e.g., from attention or feature attributions).

    Args:
        text: Input text
        word_importances: Importance score for each word
        figsize: Figure size
        cmap: Colormap for highlighting
        save_path: Path to save the plot
    """
    # Split text into words
    words = text.split()

    # Ensure word_importances has the right length
    assert len(words) == len(
        word_importances
    ), "Number of words and importances must match"

    # Normalize importances to [0, 1] for coloring
    norm_importances = (word_importances - np.min(word_importances)) / (
        np.max(word_importances) - np.min(word_importances) + 1e-8
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Configure colormap
    cmap = plt.cm.get_cmap(cmap)

    # Plot words with colored backgrounds based on importance
    x_pos = 0
    for i, (word, importance) in enumerate(zip(words, norm_importances)):
        color = cmap(importance)
        t = ax.text(
            x_pos,
            0.5,
            word + " ",
            fontsize=14,
            color="black",
            bbox=dict(facecolor=color, alpha=0.7, pad=5),
        )
        x_pos += t.get_window_extent().width

    # Configure axis
    ax.set_xlim(0, x_pos)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Word Importances")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.2)
    cbar.set_label("Normalized Importance")

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved word importances plot to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: Optional[List[Any]] = None,
    method: str = "tsne",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Visualize embeddings using dimensionality reduction.

    Args:
        embeddings: Embedding vectors (n_samples x n_dimensions)
        labels: Labels for coloring points
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Check dimensionality reduction method
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, random_state=42)
        except ImportError:
            logger.warning("scikit-learn not installed. Using PCA instead.")
            method = "pca"

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            logger.warning("umap-learn not installed. Using PCA instead.")
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)

    # Apply dimensionality reduction
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot embeddings
    if labels is not None:
        # Convert labels to integers if they're not numeric
        if not isinstance(labels[0], (int, float, np.integer, np.floating)):
            unique_labels = sorted(set(labels))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            numeric_labels = [label_to_id[label] for label in labels]
        else:
            numeric_labels = labels
            unique_labels = sorted(set(numeric_labels))

        # Plot with color-coding
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=numeric_labels,
            cmap="tab10",
            alpha=0.8,
            s=50,
        )

        # Add legend
        if (
            len(unique_labels) <= 10
        ):  # Only show legend for a reasonable number of classes
            handles, _ = scatter.legend_elements()
            plt.legend(handles, unique_labels, title="Classes")
    else:
        # Plot without color-coding
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.8, s=50)

    plt.title(f"Embeddings visualization using {method.upper()}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved embeddings visualization to {save_path}")
    else:
        plt.show()

    plt.close()
