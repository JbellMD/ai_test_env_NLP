"""
Unit tests for the text preprocessing module.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.preprocessing import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for the TextPreprocessor class."""

    def setUp(self):
        """Set up for each test."""
        self.preprocessor = TextPreprocessor()

    def test_normalize_text(self):
        """Test text normalization methods."""
        # Test basic normalization
        text = "This is a TEST with Mixed Case!!"
        normalized = self.preprocessor.normalize_text(text)
        self.assertEqual(normalized, "this is a test with mixed case!!")

    def test_remove_special_chars(self):
        """Test removal of special characters."""
        text = "Hello, world! This has @special# $characters%."
        cleaned = self.preprocessor.remove_special_chars(text)
        # Special chars should be removed but punctuation preserved by default
        self.assertNotIn("@", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("$", cleaned)
        self.assertNotIn("%", cleaned)

    def test_remove_stopwords(self):
        """Test stopword removal."""
        text = "This is a test with some stopwords."
        processed = self.preprocessor.remove_stopwords(text)
        # Common stopwords should be removed
        self.assertNotIn(" is ", " " + processed + " ")
        self.assertNotIn(" a ", " " + processed + " ")
        self.assertNotIn(" with ", " " + processed + " ")
        self.assertIn("test", processed)
        self.assertIn("stopwords", processed)

    def test_preprocess_text(self):
        """Test the complete preprocessing pipeline."""
        text = "This is a TEST with @special# characters!"
        processed = self.preprocessor.preprocess_text(
            text, normalize=True, remove_special_chars=True, remove_stopwords=True
        )
        # Result should be normalized, cleaned, and without stopwords
        self.assertEqual(processed.lower(), processed)
        self.assertNotIn("@", processed)
        self.assertNotIn("#", processed)
        self.assertNotIn(" is ", " " + processed + " ")
        self.assertNotIn(" a ", " " + processed + " ")
        self.assertIn("test", processed.lower())

    def test_batch_preprocess(self):
        """Test batch preprocessing of texts."""
        texts = ["This is a TEST!", "Another example text."]
        processed = self.preprocessor.batch_preprocess(texts)
        self.assertEqual(len(texts), len(processed))
        for text in processed:
            self.assertEqual(text.lower(), text)


if __name__ == "__main__":
    unittest.main()
