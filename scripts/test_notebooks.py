#!/usr/bin/env python
"""
Test Jupyter Notebooks

This script tests all Jupyter notebooks in the repository to ensure they run without errors.
It uses nbconvert's ExecutePreprocessor to execute the notebooks and reports any errors.
"""

import os
import sys
import argparse
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

def find_notebooks(directory):
    """Find all Jupyter notebooks in the specified directory."""
    notebooks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ipynb') and not file.endswith('-checkpoint.ipynb'):
                notebooks.append(os.path.join(root, file))
    return notebooks

def test_notebook(notebook_path, timeout=600):
    """Test a single notebook by executing all cells."""
    print(f"\nTesting notebook: {notebook_path}")
    
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Configure the preprocessor
        execute_preprocessor = ExecutePreprocessor(
            timeout=timeout,
            kernel_name='python3'
        )
        
        # Execute the notebook
        execute_preprocessor.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        print(f"✅ {notebook_path}: Passed")
        return True
    except CellExecutionError as e:
        print(f"❌ {notebook_path}: Failed")
        print(f"Error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ {notebook_path}: Error")
        print(f"Error: {str(e)}")
        return False

def main():
    """Run notebook tests."""
    parser = argparse.ArgumentParser(description='Test Jupyter notebooks')
    parser.add_argument(
        '--notebooks-dir', 
        default='notebooks',
        help='Directory containing notebooks to test (default: notebooks)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Cell execution timeout in seconds (default: 600)'
    )
    parser.add_argument(
        '--filter',
        default=None,
        help='Only test notebooks matching this substring (optional)'
    )
    
    args = parser.parse_args()
    
    # Find notebooks to test
    notebooks_dir = os.path.abspath(args.notebooks_dir)
    notebooks = find_notebooks(notebooks_dir)
    
    # Apply filter if specified
    if args.filter:
        notebooks = [nb for nb in notebooks if args.filter in os.path.basename(nb)]
    
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}" + 
              (f" matching filter '{args.filter}'" if args.filter else ""))
        return
    
    print(f"Found {len(notebooks)} notebooks to test:")
    for nb in notebooks:
        print(f"  - {os.path.relpath(nb, os.path.dirname(notebooks_dir))}")
    
    # Test each notebook
    results = {}
    for notebook in notebooks:
        results[notebook] = test_notebook(notebook, timeout=args.timeout)
    
    # Print summary
    print("\n=== Notebook Testing Summary ===")
    passed = sum(1 for success in results.values() if success)
    print(f"Passed: {passed}/{len(notebooks)}")
    
    if passed < len(notebooks):
        print("\nFailed notebooks:")
        for notebook, success in results.items():
            if not success:
                print(f"  - {os.path.relpath(notebook, os.path.dirname(notebooks_dir))}")
        sys.exit(1)

if __name__ == "__main__":
    main()
