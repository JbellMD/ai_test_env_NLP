"""
Unit tests for the API model registry.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.model_registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    """Test cases for the ModelRegistry class."""

    def setUp(self):
        """Set up for each test."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.model_dir = self.tmp_dir.name

        # Create mock model directories
        os.makedirs(os.path.join(self.model_dir, "classifier_model"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "ner_model"), exist_ok=True)

        # Create mock model metadata
        classifier_meta = {
            "task": "classification",
            "name": "Test Classifier",
            "model_type": "bert-base-uncased",
            "num_labels": 2,
            "labels": ["negative", "positive"],
            "metrics": {"accuracy": 0.85, "f1": 0.84},
        }

        ner_meta = {
            "task": "ner",
            "name": "Test NER",
            "model_type": "bert-base-ner",
            "labels": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"],
            "metrics": {"f1": 0.78},
        }

        # Write metadata files
        with open(
            os.path.join(self.model_dir, "classifier_model", "metadata.json"), "w"
        ) as f:
            json.dump(classifier_meta, f)

        with open(os.path.join(self.model_dir, "ner_model", "metadata.json"), "w") as f:
            json.dump(ner_meta, f)

        # Create registry instance
        self.registry = ModelRegistry(model_dir=self.model_dir)

    def tearDown(self):
        """Clean up after each test."""
        self.tmp_dir.cleanup()

    def test_get_available_models(self):
        """Test fetching available models from registry."""
        models = self.registry.get_available_models()

        # Check that both models are found
        self.assertEqual(len(models), 2)

        # Check model tasks
        tasks = [model["task"] for model in models]
        self.assertIn("classification", tasks)
        self.assertIn("ner", tasks)

        # Check model IDs
        ids = [model["id"] for model in models]
        self.assertIn("classifier_model", ids)
        self.assertIn("ner_model", ids)

    def test_get_models_by_task(self):
        """Test filtering models by task."""
        # Get classification models
        clf_models = self.registry.get_models_by_task("classification")
        self.assertEqual(len(clf_models), 1)
        self.assertEqual(clf_models[0]["task"], "classification")

        # Get NER models
        ner_models = self.registry.get_models_by_task("ner")
        self.assertEqual(len(ner_models), 1)
        self.assertEqual(ner_models[0]["task"], "ner")

        # Get non-existent task models
        empty_models = self.registry.get_models_by_task("translation")
        self.assertEqual(len(empty_models), 0)

    def test_get_model_metadata(self):
        """Test retrieving model metadata."""
        # Get existing model metadata
        clf_meta = self.registry.get_model_metadata("classifier_model")
        self.assertIsNotNone(clf_meta)
        self.assertEqual(clf_meta["task"], "classification")
        self.assertEqual(clf_meta["name"], "Test Classifier")

        # Try to get non-existent model metadata
        with self.assertRaises(ValueError):
            self.registry.get_model_metadata("nonexistent_model")

    def test_register_model(self):
        """Test registering a new model."""
        # Create new model metadata
        new_model_meta = {
            "task": "summarization",
            "name": "Test Summarizer",
            "model_type": "bart-large-cnn",
            "metrics": {"rouge1": 0.42, "rouge2": 0.20, "rougeL": 0.39},
        }

        # Register the model
        model_id = self.registry.register_model(
            model_path=os.path.join(self.model_dir, "summarizer_model"),
            metadata=new_model_meta,
        )

        # Check that model was registered
        self.assertEqual(model_id, "summarizer_model")

        # Check that metadata file was created
        meta_path = os.path.join(self.model_dir, "summarizer_model", "metadata.json")
        self.assertTrue(os.path.exists(meta_path))

        # Check that model appears in registry
        models = self.registry.get_available_models()
        self.assertEqual(len(models), 3)

        # Check that we can retrieve the metadata
        summ_meta = self.registry.get_model_metadata("summarizer_model")
        self.assertEqual(summ_meta["task"], "summarization")
        self.assertEqual(summ_meta["name"], "Test Summarizer")


if __name__ == "__main__":
    unittest.main()
