"""
Deployment script for NLP models.

This script provides a FastAPI server that loads NLP models for various tasks
and exposes endpoints for prediction and model management.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from src.api.app import create_app
from src.api.model_registry import ModelRegistry
from src.utils.logging_utils import get_logger

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Deploy NLP models as a web service")

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )

    # API configuration
    parser.add_argument(
        "--config", type=str, default=None, help="Path to API configuration file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Directory containing model files",
    )

    # Security options
    parser.add_argument(
        "--enable_cors",
        action="store_true",
        help="Enable CORS for frontend integration",
    )
    parser.add_argument(
        "--cors_origins",
        type=str,
        default="*",
        help="Comma-separated list of allowed origins for CORS",
    )

    # Logging options
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level",
    )

    return parser.parse_args()


def load_config(config_path=None):
    """Load API configuration from a file or use defaults."""
    default_config = {
        "host": "127.0.0.1",
        "port": 8000,
        "enable_cors": True,
        "cors_origins": ["*"],
        "enable_rate_limiting": False,
        "rate_limit": 100,
        "default_models": {
            "classification": "distilbert-base-uncased-finetuned-sst-2-english",
            "ner": "dslim/bert-base-NER",
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
            "summarization": "facebook/bart-large-cnn",
        },
        "model_dir": "./models",
        "enable_auth": False,
        "jwt_secret": "",
        "token_expire_minutes": 60,
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)

            # Merge user config with defaults
            for key, value in user_config.items():
                if key in default_config:
                    default_config[key] = value

            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}", exc_info=True)
    else:
        if config_path:
            logger.warning(f"Config file not found at {config_path}, using defaults")
        else:
            logger.info("Using default configuration")

    return default_config


def run_server(args):
    """Start the FastAPI server with the specified configuration."""
    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.host:
        config["host"] = args.host
    if args.port:
        config["port"] = args.port
    if args.enable_cors:
        config["enable_cors"] = True
    if args.cors_origins != "*":
        config["cors_origins"] = args.cors_origins.split(",")
    if args.model_dir:
        config["model_dir"] = args.model_dir

    # Create FastAPI app
    app = create_app(config)

    # Start the server
    logger.info(f"Starting NLP API server at http://{config['host']}:{config['port']}")
    uvicorn.run(app, host=config["host"], port=config["port"])


if __name__ == "__main__":
    args = parse_args()
    run_server(args)
