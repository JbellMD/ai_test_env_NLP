"""
Text summarization models and utilities.

This module provides classes and functions for both extractive and
abstractive text summarization tasks.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Dict, List, Optional, Union, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextSummarizer:
    """
    Wrapper class for transformer-based text summarization models.
    
    This class provides a unified interface for abstractive summarization
    using sequence-to-sequence transformer models.
    """
    
    def __init__(self, 
                 model_name_or_path: str,
                 device: Optional[str] = None):
        """
        Initialize the summarizer with a pre-trained model.
        
        Args:
            model_name_or_path: Hugging Face model name or path to local model
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name_or_path
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
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
        Train the summarization model.
        
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
            'rouge_scores': []
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
                    
                    if 'rouge' in eval_results:
                        history['rouge_scores'].append(eval_results['rouge'])
                    
                    # Print progress
                    print(f"Epoch {epoch+1}/{num_epochs} | Step {step} | "
                          f"Train Loss: {loss.item():.4f} | "
                          f"Eval Loss: {eval_results['loss']:.4f}")
                    
                    # Return to training mode
                    self.model.train()
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} completed | "
                  f"Average Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        # Save the model if path provided
        if save_path:
            self.save(save_path)
        
        return history
    
    def evaluate(self, dataloader, tokenizer=None, compute_rouge=False):
        """
        Evaluate the summarization model.
        
        Args:
            dataloader: DataLoader for evaluation data
            tokenizer: Tokenizer for decoding predictions
            compute_rouge: Whether to compute ROUGE scores
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        
        # ROUGE evaluation requires decoded predictions
        all_decoded_preds = []
        all_decoded_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Generate summaries if ROUGE computation is needed
                if compute_rouge and tokenizer:
                    # Generate predictions
                    generated_ids = self.model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    # Decode predictions and labels
                    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                    
                    all_decoded_preds.extend(decoded_preds)
                    all_decoded_labels.extend(decoded_labels)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(dataloader)
        }
        
        # Compute ROUGE if needed
        if compute_rouge and tokenizer:
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
                
                for pred, label in zip(all_decoded_preds, all_decoded_labels):
                    score = scorer.score(label, pred)
                    rouge_scores['rouge1'].append(score['rouge1'].fmeasure)
                    rouge_scores['rouge2'].append(score['rouge2'].fmeasure)
                    rouge_scores['rougeL'].append(score['rougeL'].fmeasure)
                
                # Average scores
                metrics['rouge'] = {
                    'rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']),
                    'rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']),
                    'rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
                }
            except ImportError:
                print("Warning: rouge_score package not found. ROUGE metrics not computed.")
        
        return metrics
    
    def summarize(self, 
                 encoded_inputs: Dict[str, torch.Tensor],
                 max_length: int = 128,
                 min_length: int = 30,
                 num_beams: int = 4,
                 early_stopping: bool = True,
                 **generate_kwargs):
        """
        Generate summaries for encoded inputs.
        
        Args:
            encoded_inputs: Encoded inputs from tokenizer
            max_length: Maximum length of generated summaries
            min_length: Minimum length of generated summaries
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop when all beams are finished
            **generate_kwargs: Additional arguments for generate method
            
        Returns:
            Generated summary IDs
        """
        self.model.eval()
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in encoded_inputs.items() 
                 if k in ['input_ids', 'attention_mask']}
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                **generate_kwargs
            )
        
        return summary_ids
    
    def summarize_text(self, 
                      texts: Union[str, List[str]], 
                      tokenizer,
                      max_length: int = 128,
                      min_length: int = 30,
                      batch_size: int = 4,
                      **generate_kwargs):
        """
        Summarize text inputs.
        
        Args:
            texts: Input text or list of texts
            tokenizer: Tokenizer to use
            max_length: Maximum length of generated summaries
            min_length: Minimum length of generated summaries
            batch_size: Batch size for processing
            **generate_kwargs: Additional arguments for generate method
            
        Returns:
            List of summaries
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        all_summaries = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,  # Maximum model input length
                return_tensors='pt'
            )
            
            # Generate summaries
            summary_ids = self.summarize(
                inputs,
                max_length=max_length,
                min_length=min_length,
                **generate_kwargs
            )
            
            # Decode summaries
            batch_summaries = tokenizer.batch_decode(
                summary_ids, 
                skip_special_tokens=True
            )
            
            all_summaries.extend(batch_summaries)
        
        # If input was a single text, return a single summary
        if single_input:
            return all_summaries[0]
        
        return all_summaries
    
    def save(self, path: str):
        """
        Save the model.
        
        Args:
            path: Directory path to save the model
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.model.save_pretrained(path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load a model from a directory.
        
        Args:
            path: Directory path containing the saved model
            device: Device to load the model on
            
        Returns:
            TextSummarizer instance
        """
        # Create instance
        instance = cls(
            model_name_or_path=path,
            device=device
        )
        
        return instance


class ExtractiveSummarizer:
    """
    Extractive text summarization using non-neural methods.
    
    This class implements extractive summarization techniques based on
    sentence ranking algorithms.
    """
    
    def __init__(self, method: str = 'tfidf'):
        """
        Initialize the extractive summarizer.
        
        Args:
            method: Summarization method ('tfidf', 'textrank', 'lexrank')
        """
        self.method = method.lower()
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def summarize(self, 
                 text: str, 
                 ratio: float = 0.3, 
                 max_sentences: Optional[int] = None):
        """
        Generate an extractive summary.
        
        Args:
            text: Input text to summarize
            ratio: Proportion of sentences to include in summary
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Extractive summary
        """
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return ""
        
        # Calculate number of sentences for summary
        n_sentences = len(sentences)
        
        if max_sentences is not None:
            n_summary = min(max_sentences, int(ratio * n_sentences))
        else:
            n_summary = int(ratio * n_sentences)
        
        n_summary = max(1, n_summary)  # At least 1 sentence
        
        # Get ranked sentence indices
        if self.method == 'tfidf':
            ranked_indices = self._tfidf_ranking(sentences)
        elif self.method == 'textrank':
            ranked_indices = self._textrank_ranking(sentences)
        elif self.method == 'lexrank':
            ranked_indices = self._lexrank_ranking(sentences)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Select top sentences in original order
        selected_indices = sorted(ranked_indices[:n_summary])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        # Join sentences into a summary
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def _tfidf_ranking(self, sentences: List[str]) -> List[int]:
        """
        Rank sentences using TF-IDF and cosine similarity.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of sentence indices ranked by importance
        """
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Transform sentences to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence similarities (sentence vs. all sentences)
        similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Calculate sentence scores (sum of similarities)
        scores = similarities.sum(axis=1)
        
        # Rank sentences by score
        ranked_indices = scores.argsort()[::-1].tolist()
        
        return ranked_indices
    
    def _textrank_ranking(self, sentences: List[str]) -> List[int]:
        """
        Rank sentences using TextRank algorithm.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of sentence indices ranked by importance
        """
        # Create TF-IDF vectorizer for sentence similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Transform sentences to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence similarities
        similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create similarity graph and apply PageRank
        nx_graph = self._create_graph_from_similarities(similarities)
        scores = self._apply_pagerank(nx_graph)
        
        # Rank sentences by score
        ranked_indices = sorted(((score, i) for i, score in enumerate(scores)), reverse=True)
        ranked_indices = [i for _, i in ranked_indices]
        
        return ranked_indices
    
    def _lexrank_ranking(self, sentences: List[str]) -> List[int]:
        """
        Rank sentences using LexRank algorithm.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of sentence indices ranked by importance
        """
        # Similar to TextRank but with different similarity measure
        # For this implementation, we'll use a simplified version
        return self._textrank_ranking(sentences)
    
    def _create_graph_from_similarities(self, similarities, threshold: float = 0.3):
        """
        Create a graph from similarities matrix.
        
        Args:
            similarities: Sentence similarity matrix
            threshold: Similarity threshold for edge creation
            
        Returns:
            NetworkX graph
        """
        import networkx as nx
        
        # Create a graph
        graph = nx.Graph()
        
        # Add nodes
        for i in range(len(similarities)):
            graph.add_node(i)
        
        # Add edges
        for i in range(len(similarities)):
            for j in range(len(similarities)):
                if i != j and similarities[i, j] >= threshold:
                    graph.add_edge(i, j, weight=similarities[i, j])
        
        return graph
    
    def _apply_pagerank(self, graph, alpha: float = 0.85, max_iter: int = 100):
        """
        Apply PageRank algorithm to graph.
        
        Args:
            graph: NetworkX graph
            alpha: Damping factor
            max_iter: Maximum iterations
            
        Returns:
            PageRank scores
        """
        import networkx as nx
        
        # Apply PageRank
        pagerank = nx.pagerank(
            graph,
            alpha=alpha,
            max_iter=max_iter
        )
        
        # Extract scores
        scores = [pagerank[i] for i in range(len(pagerank))]
        
        return scores


class ControlledSummarizer:
    """
    Controlled text summarization with user-specified parameters.
    
    This class extends basic summarization with control over
    summary length, focus, and style.
    """
    
    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        """
        Initialize the controlled summarizer.
        
        Args:
            model_name_or_path: Hugging Face model name or path
            device: Device to use
        """
        self.summarizer = TextSummarizer(
            model_name_or_path=model_name_or_path,
            device=device
        )
    
    def summarize(self, 
                 text: str, 
                 tokenizer,
                 length: str = 'medium',  # 'short', 'medium', 'long'
                 focus: Optional[str] = None,  # Focus keyword or topic
                 style: str = 'default'):  # 'default', 'simple', 'detailed'
        """
        Generate a controlled summary.
        
        Args:
            text: Input text to summarize
            tokenizer: Tokenizer to use
            length: Desired summary length
            focus: Keyword or topic to focus on
            style: Summary style
            
        Returns:
            Controlled summary
        """
        # Set length parameters
        if length == 'short':
            max_length = 64
            min_length = 10
        elif length == 'medium':
            max_length = 128
            min_length = 30
        elif length == 'long':
            max_length = 256
            min_length = 64
        else:
            max_length = 128
            min_length = 30
        
        # Construct prompt with control parameters
        prompt = text
        
        if focus is not None:
            prompt = f"Summarize the following text, focusing on {focus}: {text}"
        
        if style == 'simple':
            prompt = f"Summarize the following text in simple language: {prompt}"
        elif style == 'detailed':
            prompt = f"Provide a detailed summary of the following text: {prompt}"
        
        # Tokenize with control prompt
        inputs = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )
        
        # Generate summary
        summary_ids = self.summarizer.summarize(
            inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary


# Factory functions
def create_abstractive_summarizer(model_name: str, **kwargs):
    """
    Create an abstractive summarizer based on model name.
    
    Args:
        model_name: Model name or path
        **kwargs: Additional arguments for TextSummarizer
        
    Returns:
        TextSummarizer instance
    """
    return TextSummarizer(
        model_name_or_path=model_name,
        **kwargs
    )


def create_extractive_summarizer(method: str = 'tfidf', **kwargs):
    """
    Create an extractive summarizer.
    
    Args:
        method: Summarization method
        **kwargs: Additional arguments for ExtractiveSummarizer
        
    Returns:
        ExtractiveSummarizer instance
    """
    return ExtractiveSummarizer(
        method=method,
        **kwargs
    )


def create_controlled_summarizer(model_name: str, **kwargs):
    """
    Create a controlled summarizer.
    
    Args:
        model_name: Model name or path
        **kwargs: Additional arguments for ControlledSummarizer
        
    Returns:
        ControlledSummarizer instance
    """
    return ControlledSummarizer(
        model_name_or_path=model_name,
        **kwargs
    )
