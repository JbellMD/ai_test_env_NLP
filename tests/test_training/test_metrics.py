"""
Unit tests for the metrics calculation module.
"""
import unittest
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, rouge_metrics
)


class TestMetrics(unittest.TestCase):
    """Test cases for the metrics calculation functions."""

    def test_classification_metrics(self):
        """Test classification metric calculations."""
        # Sample data
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 1, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        # Check that metrics are in expected range [0, 1]
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)
        
        self.assertGreaterEqual(prec, 0)
        self.assertLessEqual(prec, 1)
        
        self.assertGreaterEqual(rec, 0)
        self.assertLessEqual(rec, 1)
        
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        
        # Expected accuracy: 3/6 = 0.5
        self.assertAlmostEqual(acc, 0.5, places=5)
    
    def test_classification_report_format(self):
        """Test classification report formatting."""
        # Sample data
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 1, 1]
        
        # Generate report
        report = classification_report(y_true, y_pred)
        
        # Check that report has expected structure
        self.assertIsInstance(report, dict)
        
        # Classes should be in report
        for cls in sorted(set(y_true)):
            self.assertIn(str(cls), report)
            
            # Each class should have precision, recall, f1-score, and support
            self.assertIn('precision', report[str(cls)])
            self.assertIn('recall', report[str(cls)])
            self.assertIn('f1-score', report[str(cls)])
            self.assertIn('support', report[str(cls)])
        
        # Check aggregates
        self.assertIn('accuracy', report)
        self.assertIn('macro avg', report)
        self.assertIn('weighted avg', report)
    
    def test_rouge_metrics(self):
        """Test ROUGE metric calculations."""
        # Sample data
        summaries = [
            "This is a test summary.",
            "Another sample summary with more words."
        ]
        references = [
            "This is a reference summary.",
            "Another sample reference with additional words."
        ]
        
        # Calculate ROUGE scores
        rouge_scores = rouge_metrics(summaries, references)
        
        # Check that scores are returned
        self.assertIn('rouge1', rouge_scores)
        self.assertIn('rouge2', rouge_scores)
        self.assertIn('rougeL', rouge_scores)
        
        # Check that scores are in expected range [0, 1]
        self.assertGreaterEqual(rouge_scores['rouge1'], 0)
        self.assertLessEqual(rouge_scores['rouge1'], 1)
        
        self.assertGreaterEqual(rouge_scores['rouge2'], 0)
        self.assertLessEqual(rouge_scores['rouge2'], 1)
        
        self.assertGreaterEqual(rouge_scores['rougeL'], 0)
        self.assertLessEqual(rouge_scores['rougeL'], 1)
    
    def test_empty_inputs(self):
        """Test metrics functions with empty inputs."""
        # Edge case: empty lists
        y_true = []
        y_pred = []
        
        # This should not raise errors but return default values or zeros
        acc = accuracy_score(y_true, y_pred)
        self.assertEqual(acc, 0.0)  # Convention: accuracy of empty set is 0
        
        # For ROUGE, empty inputs should be handled gracefully
        with self.assertRaises(ValueError):
            # ROUGE typically requires non-empty inputs
            rouge_metrics([], [])


if __name__ == "__main__":
    unittest.main()
