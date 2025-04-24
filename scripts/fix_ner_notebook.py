#!/usr/bin/env python
"""
Script to create a fixed version of the NER notebook with robust dataset access.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def fix_ner_notebook():
    """
    Create a fixed version of the NER notebook with robust dataset access.
    """
    # Paths
    original_notebook = os.path.join(project_root, "notebooks", "02_named_entity_recognition_demo.ipynb")
    fixed_notebook = os.path.join(project_root, "notebooks", "02_named_entity_recognition_demo_fixed.ipynb")
    
    print(f"Loading original notebook: {original_notebook}")
    
    # Load the original notebook
    with open(original_notebook, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Find and fix the cell with dataset access
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Look for the problematic cell
            if "dataset = dataset_loader.load_huggingface_dataset" in source and "example['tokens']" in source:
                print(f"Found problematic cell at index {i}")
                
                # Replace with robust code
                new_source = [
                    "# Load the ConLL-2003 dataset\n",
                    "dataset = dataset_loader.load_huggingface_dataset(\n",
                    "    dataset_name=DATASET_NAME,\n",
                    "    text_column=\"tokens\",\n",
                    "    label_column=\"ner_tags\"\n",
                    ")\n",
                    "\n",
                    "# Display dataset information\n",
                    "print(f\"Dataset: {DATASET_NAME}\")\n",
                    "print(f\"Number of splits: {len(dataset.keys())}\")\n",
                    "for split in dataset.keys():\n",
                    "    print(f\"  {split}: {len(dataset[split])} examples\")\n",
                    "\n",
                    "# Show example data with ultra-robust access pattern\n",
                    "print(\"\\nExample data:\")\n",
                    "for i, example in enumerate(dataset[\"train\"][:2]):\n",
                    "    print(f\"Example {i+1}:\")\n",
                    "    \n",
                    "    # Ultra-robust access pattern that handles string examples\n",
                    "    if isinstance(example, dict) and 'tokens' in example:\n",
                    "        # Dictionary with tokens key\n",
                    "        tokens = example['tokens']\n",
                    "        if isinstance(tokens, list) and len(tokens) > 0:\n",
                    "            print(f\"  Tokens: {tokens[:10]}...\")\n",
                    "            \n",
                    "        # Get NER tags if available\n",
                    "        if 'ner_tags' in example and isinstance(example['ner_tags'], list):\n",
                    "            print(f\"  NER tags: {example['ner_tags'][:10]}...\")\n",
                    "            \n",
                    "    elif isinstance(example, dict):\n",
                    "        # Dictionary without expected keys\n",
                    "        print(f\"  Available keys: {list(example.keys())}\")\n",
                    "        \n",
                    "    elif isinstance(example, str):\n",
                    "        # String example (this might be a JSON string that needs parsing)\n",
                    "        print(f\"  Example is a string. First 50 chars: {example[:50]}...\")\n",
                    "        \n",
                    "        # Try to parse as JSON\n",
                    "        try:\n",
                    "            import json\n",
                    "            parsed = json.loads(example)\n",
                    "            if isinstance(parsed, dict):\n",
                    "                print(f\"  Parsed JSON keys: {list(parsed.keys())}\")\n",
                    "                if 'tokens' in parsed:\n",
                    "                    print(f\"  Tokens from JSON: {parsed['tokens'][:10]}...\")\n",
                    "                if 'ner_tags' in parsed:\n",
                    "                    print(f\"  NER tags from JSON: {parsed['ner_tags'][:10]}...\")\n",
                    "        except json.JSONDecodeError:\n",
                    "            # Not valid JSON, try other approaches\n",
                    "            # Just print it directly\n",
                    "            print(f\"  (Not valid JSON) Content: {example}\")\n",
                    "    else:\n",
                    "        # Some other type\n",
                    "        print(f\"  Example type: {type(example)}\")\n"
                ]
                
                notebook["cells"][i]["source"] = new_source
                print("Cell updated with robust dataset access code")
    
    # Write the fixed notebook
    with open(fixed_notebook, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Fixed notebook saved to: {fixed_notebook}")
    print("You can now run the fixed notebook to see if the dataset access issues are resolved.")

if __name__ == "__main__":
    fix_ner_notebook()
