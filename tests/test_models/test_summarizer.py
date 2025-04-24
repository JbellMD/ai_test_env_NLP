"""
Unit tests for the text summarization model.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.summarizer import ExtractiveSummarizer, TextSummarizer


class TestTextSummarizer(unittest.TestCase):
    """Test cases for the TextSummarizer class."""

    @classmethod
    def setUpClass(cls):
        """Set up once before all tests."""
        # Skip full model loading tests if no GPU available to speed up testing
        cls.skip_model_tests = not torch.cuda.is_available()

        if not cls.skip_model_tests:
            # Use a small, fast model for testing
            cls.model_name = "facebook/bart-large-cnn"

    def setUp(self):
        """Set up for each test."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.model_path = self.tmp_dir.name

    def tearDown(self):
        """Clean up after each test."""
        self.tmp_dir.cleanup()

    def test_model_initialization(self):
        """Test that model can be initialized without errors."""
        if self.skip_model_tests:
            self.skipTest("Skipping model tests because no GPU is available")

        # Initialize model
        summarizer = TextSummarizer(model_name=self.model_name)

        # Basic checks
        self.assertIsNotNone(summarizer.model)
        self.assertIsNotNone(summarizer.config)

    def test_summarization(self):
        """Test summarization functionality."""
        if self.skip_model_tests:
            self.skipTest("Skipping model tests because no GPU is available")

        # Initialize model
        summarizer = TextSummarizer(model_name=self.model_name)

        # Test data - a long paragraph
        text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
        concerned with the interactions between computers and human language, in particular how to program computers 
        to process and analyze large amounts of natural language data. The goal is a computer capable of understanding 
        the contents of documents, including the contextual nuances of the language within them. The technology can then 
        accurately extract information and insights contained in the documents as well as categorize and organize the 
        documents themselves. Challenges in natural language processing frequently involve speech recognition, natural 
        language understanding, and natural language generation.
        """

        # Generate summary
        summary = summarizer.summarize_text([text], max_length=50, min_length=10)[0]

        # Check that summary is shorter than original text
        self.assertLess(len(summary.split()), len(text.split()))
        self.assertGreater(len(summary), 0)

    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        if self.skip_model_tests:
            self.skipTest("Skipping model tests because no GPU is available")

        # Initialize model
        summarizer = TextSummarizer(model_name=self.model_name)

        # Save model
        save_path = os.path.join(self.model_path, "test_summarizer")
        summarizer.save(save_path)

        # Check that files were created
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, "config.json")))

        # Load model
        loaded_summarizer = TextSummarizer.load(save_path)

        # Check that loaded model has the same properties
        self.assertEqual(
            loaded_summarizer.model.config.model_type,
            summarizer.model.config.model_type,
        )


class TestExtractiveSummarizer(unittest.TestCase):
    """Test cases for the ExtractiveSummarizer class."""

    def test_textrank_summarization(self):
        """Test summarization using TextRank algorithm."""
        summarizer = ExtractiveSummarizer(method="textrank")

        # Test data - multiple sentences
        text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. 
        It is concerned with the interactions between computers and human language. 
        The goal is a computer capable of understanding documents, including contextual nuances. 
        Challenges in NLP frequently involve speech recognition, natural language understanding, and generation.
        """

        # Generate summary
        summary = summarizer.summarize(text, ratio=0.5)

        # Check that summary is shorter than original text
        self.assertLess(len(summary.split()), len(text.split()))
        self.assertGreater(len(summary), 0)

    def test_lsa_summarization(self):
        """Test summarization using LSA algorithm."""
        summarizer = ExtractiveSummarizer(method="lsa")

        # Test data - multiple sentences
        text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. 
        It is concerned with the interactions between computers and human language. 
        The goal is a computer capable of understanding documents, including contextual nuances. 
        Challenges in NLP frequently involve speech recognition, natural language understanding, and generation.
        """

        # Generate summary
        summary = summarizer.summarize(text, ratio=0.5)

        # Check that summary is shorter than original text
        self.assertLess(len(summary.split()), len(text.split()))
        self.assertGreater(len(summary), 0)


if __name__ == "__main__":
    unittest.main()
