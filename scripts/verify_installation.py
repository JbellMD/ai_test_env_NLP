#!/usr/bin/env python
"""
Verify NLP Toolkit Installation

This script verifies that the NLP Toolkit has been installed correctly
and all required dependencies are available.
"""

import importlib
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add the project root to path so we can import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Required modules to check
REQUIRED_MODULES = [
    # Core dependencies
    "torch",
    "transformers", 
    "datasets",
    "sklearn",
    "numpy", 
    "pandas",
    "nltk",
    
    # API dependencies
    "fastapi",
    "pydantic",
    "uvicorn",
    
    # Additional NLP libraries
    "spacy",
    "rouge_score",
    "seqeval",
    
    # Visualization
    "matplotlib",
    "seaborn"
]

# Project modules to check
PROJECT_MODULES = [
    "src.data.preprocessing",
    "src.models.summarizer",
    "src.models.classifier",
    "src.training.metrics",
    "src.api.model_registry",
    "src.utils.logging_utils"
]

def check_module(module_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if a module can be imported and get its version if available.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        Tuple containing:
        - Boolean indicating if module was successfully imported
        - Version string if available, None otherwise
        - Error message if import failed, None otherwise
    """
    try:
        module = importlib.import_module(module_name.split('.')[0])
        
        # Get version if available
        version = None
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                break
                
        # For nested imports, check if the full path is importable
        if '.' in module_name:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                return False, version, f"Could not import {module_name}: {str(e)}"
                
        return True, version, None
    except ImportError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"

def verify_installation() -> Dict[str, Dict]:
    """
    Verify the installation of all required modules.
    
    Returns:
        Dictionary with verification results
    """
    results = {"external_modules": {}, "project_modules": {}}
    all_external_passed = True
    all_project_passed = True
    
    # Check external dependencies
    for module_name in REQUIRED_MODULES:
        success, version, error = check_module(module_name)
        
        results["external_modules"][module_name] = {
            "installed": success,
            "version": version,
            "error": error
        }
        
        if not success:
            all_external_passed = False
    
    # Check project modules
    for module_name in PROJECT_MODULES:
        success, version, error = check_module(module_name)
        
        results["project_modules"][module_name] = {
            "installed": success,
            "version": version,
            "error": error
        }
        
        if not success:
            all_project_passed = False
    
    return {
        "all_external_passed": all_external_passed,
        "all_project_passed": all_project_passed, 
        "external_modules": results["external_modules"],
        "project_modules": results["project_modules"]
    }

def print_results(results: Dict) -> None:
    """
    Print verification results in a readable format.
    
    Args:
        results: Results dictionary from verify_installation
    """
    print("\n=== NLP Toolkit Installation Verification ===\n")
    
    # External dependencies
    print("External Dependencies:")
    print("---------------------")
    
    if results["all_external_passed"]:
        print("✅ All external dependencies successfully installed!\n")
    else:
        print("❌ Some external dependencies are missing or could not be imported.\n")
    
    for module_name, module_info in results["external_modules"].items():
        if module_info["installed"]:
            version_str = f" (v{module_info['version']})" if module_info["version"] else ""
            print(f"✅ {module_name}{version_str}")
        else:
            print(f"❌ {module_name}: {module_info['error']}")
    
    # Project modules
    print("\nProject Modules:")
    print("---------------")
    
    if results["all_project_passed"]:
        print("✅ All project modules successfully imported!\n")
    else:
        print("❌ Some project modules could not be imported.\n")
        print("   This is expected if running from source without installing the package.\n")
        print("   To properly test installation, create a virtual environment and install with:\n")
        print("   pip install -e .\n")
    
    for module_name, module_info in results["project_modules"].items():
        if module_info["installed"]:
            print(f"✅ {module_name}")
        else:
            print(f"❌ {module_name}: {module_info['error']}")
    
    print("\n")

def main():
    """Run the installation verification."""
    results = verify_installation()
    print_results(results)
    
    # Only exit with error if external dependencies fail
    if not results["all_external_passed"]:
        print("ERROR: External dependencies missing. Please install required packages.")
        sys.exit(1)
    
    if not results["all_project_passed"]:
        print("WARNING: Project modules could not be imported from the current environment.")
        print("This may be normal if running from source without installing the package.")

if __name__ == "__main__":
    main()
