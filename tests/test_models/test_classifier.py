"""
Unit tests for the text classification model.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.classifier import TransformerClassifier


class TestTransformerClassifier(unittest.TestCase):
    """Test cases for the TransformerClassifier class."""

    @classmethod
    def setUpClass(cls):
        """Set up once before all tests."""
        # Skip full model loading tests if no GPU available to speed up testing
        cls.skip_model_tests = not torch.cuda.is_available()

        if not cls.skip_model_tests:
            # Use a small, fast model for testing
            cls.model_name = "distilbert-base-uncased"
            cls.num_labels = 2

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
        classifier = TransformerClassifier(
            model_name=self.model_name, num_labels=self.num_labels
        )

        # Basic checks
        self.assertIsNotNone(classifier.model)
        self.assertIsNotNone(classifier.config)
        self.assertEqual(classifier.num_labels, self.num_labels)

    def test_prediction_shape(self):
        """Test that prediction outputs have the expected shape."""
        if self.skip_model_tests:
            self.skipTest("Skipping model tests because no GPU is available")

        # Initialize model
        classifier = TransformerClassifier(
            model_name=self.model_name, num_labels=self.num_labels
        )

        # Test data
        texts = ["This is a positive review.", "This is a negative review."]

        # Make predictions
        predictions = classifier.predict(texts)

        # Check shape
        self.assertEqual(len(predictions), len(texts))
        for pred in predictions:
            self.assertIsInstance(pred, int)
            self.assertGreaterEqual(pred, 0)
            self.assertLess(pred, self.num_labels)

    def test_predict_proba(self):
        """Test probability predictions."""
        if self.skip_model_tests:
            self.skipTest("Skipping model tests because no GPU is available")

        # Initialize model
        classifier = TransformerClassifier(
            model_name=self.model_name, num_labels=self.num_labels
        )

        # Test data
        texts = ["This is a positive review.", "This is a negative review."]

        # Make predictions
        probas = classifier.predict_proba(texts)

        # Check shape and probabilities
        self.assertEqual(len(probas), len(texts))
        for proba in probas:
            self.assertEqual(len(proba), self.num_labels)
            self.assertAlmostEqual(sum(proba), 1.0, places=5)
            for p in proba:
                self.assertGreaterEqual(p, 0)
                self.assertLessEqual(p, 1)

    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        if self.skip_model_tests:
            self.skipTest("Skipping model tests because no GPU is available")

        # Initialize model
        classifier = TransformerClassifier(
            model_name=self.model_name, num_labels=self.num_labels
        )

        # Save model
        save_path = os.path.join(self.model_path, "test_model")
        classifier.save(save_path)

        # Check that files were created
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, "config.json")))

        # Load model
        loaded_classifier = TransformerClassifier.load(save_path)

        # Check that loaded model has the same properties
        self.assertEqual(loaded_classifier.num_labels, classifier.num_labels)
        self.assertEqual(loaded_classifier.config.id2label, classifier.config.id2label)

        # Test predictions with loaded model
        texts = ["This is a test."]
        original_preds = classifier.predict(texts)
        loaded_preds = loaded_classifier.predict(texts)

        # Check that predictions are the same
        self.assertEqual(original_preds, loaded_preds)


if __name__ == "__main__":
    unittest.main()
