"""
Simple test script for the NLP Toolkit.

This script tests the core functionality of the NLP toolkit by:
1. Preprocessing some sample text
2. Running a simple text classification task
3. Testing the summarization module
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import TextPreprocessor
from src.models.summarizer import ExtractiveSummarizer

def test_preprocessing():
    """Test the text preprocessing capabilities."""
    print("\n=== Testing Text Preprocessing ===")
    
    # Sample text
    text = "This is a TEST with @special# characters! And some repeated words words."
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=True, 
        remove_punctuation=True,
        remove_stopwords=True
    )
    
    # Test individual preprocessing methods
    print(f"Original text: '{text}'")
    print(f"Normalized text: '{preprocessor.normalize_text(text)}'")
    print(f"Text without special chars: '{preprocessor.remove_special_chars(text)}'")
    print(f"Text without stopwords: '{preprocessor.remove_stopwords(text)}'")
    
    # Test complete preprocessing
    processed = preprocessor.preprocess(text)
    print(f"Fully preprocessed text: '{processed}'")

def test_summarization():
    """Test the summarization capabilities."""
    print("\n=== Testing Text Summarization ===")
    
    # Sample text for summarization
    text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.
    It is concerned with the interactions between computers and human language.
    The goal is a computer capable of understanding documents, including contextual nuances.
    Challenges in NLP frequently involve speech recognition, natural language understanding, and generation.
    Machine learning and deep learning methods are widely used for NLP tasks today.
    Transformer models like BERT and GPT have revolutionized the field in recent years.
    These models can perform tasks ranging from sentiment analysis to question answering.
    Transfer learning has enabled models trained on large corpora to be fine-tuned for specific tasks.
    """
    
    # Test different summarization methods
    for method in ["tfidf", "textrank", "lexrank", "lsa"]:
        summarizer = ExtractiveSummarizer(method=method)
        summary = summarizer.summarize(text, ratio=0.3)
        print(f"\n{method.upper()} Summary:")
        print(f"'{summary}'")

if __name__ == "__main__":
    test_preprocessing()
    test_summarization()
