"""
Setup script for the NLP package.

This script configures the package installation, dependencies, and metadata
for distribution as a Python package.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="nlp_toolkit",
    version="0.1.0",
    author="NLP Project Team",
    author_email="info@nlpproject.org",
    description="A comprehensive NLP toolkit for multiple tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp_toolkit",
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'nlp-train=scripts.train:main',
            'nlp-evaluate=scripts.evaluate:main',
            'nlp-deploy=scripts.deploy:main',
        ],
    },
    include_package_data=True,
    package_data={
        'src': ['configs/*.json', 'configs/model_configs/*.json'],
    },
)
