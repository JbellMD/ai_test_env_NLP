"""
Unit tests for the data loader module.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.data_loader import NLPDatasetLoader
from src.data.preprocessing import TextPreprocessor


class TestNLPDatasetLoader(unittest.TestCase):
    """Test cases for the NLPDatasetLoader class."""

    def setUp(self):
        """Set up for each test."""
        self.preprocessor = TextPreprocessor()
        self.loader = NLPDatasetLoader(self.preprocessor)

        # Create temporary test data
        self.temp_dir = tempfile.TemporaryDirectory()

        # Test CSV file
        self.csv_data = pd.DataFrame(
            {"text": ["This is a test.", "Another test document."], "label": [0, 1]}
        )
        self.csv_path = os.path.join(self.temp_dir.name, "test_data.csv")
        self.csv_data.to_csv(self.csv_path, index=False)

        # Test JSON file
        self.json_data = [
            {"text": "Json test document", "label": 0},
            {"text": "Another json document", "label": 1},
        ]
        self.json_path = os.path.join(self.temp_dir.name, "test_data.json")
        with open(self.json_path, "w") as f:
            json.dump(self.json_data, f)

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_load_from_csv(self):
        """Test loading data from CSV file."""
        dataset = self.loader.load_from_csv(
            self.csv_path, text_column="text", label_column="label"
        )

        # Check dataset structure
        self.assertEqual(len(dataset), len(self.csv_data))
        self.assertIn("text", dataset.column_names)
        self.assertIn("label", dataset.column_names)

        # Check data content
        for i, row in enumerate(dataset):
            self.assertEqual(row["text"], self.csv_data.iloc[i]["text"])
            self.assertEqual(row["label"], self.csv_data.iloc[i]["label"])

    def test_load_from_json(self):
        """Test loading data from JSON file."""
        dataset = self.loader.load_from_json(
            self.json_path, text_key="text", label_key="label"
        )

        # Check dataset structure
        self.assertEqual(len(dataset), len(self.json_data))
        self.assertIn("text", dataset.column_names)
        self.assertIn("label", dataset.column_names)

        # Check data content
        for i, row in enumerate(dataset):
            self.assertEqual(row["text"], self.json_data[i]["text"])
            self.assertEqual(row["label"], self.json_data[i]["label"])

    def test_preprocess_dataset(self):
        """Test dataset preprocessing functionality."""
        # Load dataset
        dataset = self.loader.load_from_csv(
            self.csv_path, text_column="text", label_column="label"
        )

        # Create a simple preprocessing function for testing
        def test_preprocessing_fn(examples):
            examples["processed_text"] = [text.upper() for text in examples["text"]]
            return examples

        # Apply preprocessing
        processed_dataset = self.loader.preprocess_dataset(
            dataset, preprocessing_fn=test_preprocessing_fn
        )

        # Check that preprocessing was applied
        self.assertIn("processed_text", processed_dataset.column_names)
        for i, row in enumerate(processed_dataset):
            self.assertEqual(
                row["processed_text"], self.csv_data.iloc[i]["text"].upper()
            )


if __name__ == "__main__":
    unittest.main()
