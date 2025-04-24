"""
Logging utilities for NLP tasks.

This module provides functions for setting up and configuring
logging across the application.
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Args:
        name: Logger name (typically module name)
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add stream handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_file_logging(
    logger: logging.Logger, log_dir: str, filename: Optional[str] = None
) -> str:
    """
    Set up file logging for the specified logger.

    Args:
        logger: Logger instance
        log_dir: Directory for log files
        filename: Log file name (default: auto-generated based on timestamp)

    Returns:
        Path to the log file
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{logger.name}_{timestamp}.log"

    log_path = os.path.join(log_dir, filename)

    # Add file handler
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to file: {log_path}")

    return log_path


class JsonLogger:
    """Logger that stores structured records as JSON."""

    def __init__(self, log_dir: str, filename: Optional[str] = None):
        """
        Initialize the JSON logger.

        Args:
            log_dir: Directory for log files
            filename: Log file name (default: auto-generated based on timestamp)
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"json_log_{timestamp}.jsonl"

        self.log_path = os.path.join(log_dir, filename)
        self.logger = get_logger(f"jsonlogger.{filename}")

        self.logger.info(f"JSON logging to file: {self.log_path}")

    def log(self, record: Dict[str, Any]) -> None:
        """
        Log a record as JSON.

        Args:
            record: Record to log (dictionary)
        """
        # Add timestamp if not present
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        # Write to file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric values
            step: Training step or epoch
            prefix: Prefix for metric names
        """
        record = {"type": "metrics", "step": step}

        # Add prefixed metrics
        for name, value in metrics.items():
            key = f"{prefix}{name}" if prefix else name
            record[key] = value

        self.log(record)

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log model or training parameters.

        Args:
            parameters: Dictionary of parameter values
        """
        record = {"type": "parameters", **parameters}

        self.log(record)

    def log_artifact(
        self, artifact_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a reference to an artifact.

        Args:
            artifact_path: Path to the artifact
            metadata: Additional metadata about the artifact
        """
        record: Dict[str, Any] = {"type": "artifact", "path": artifact_path}

        if metadata:
            record["metadata"] = metadata

        self.log(record)

class ExperimentTracker:
    """Track experiments with metrics, parameters, and artifacts."""

    def __init__(
        self,
        experiment_name: str,
        run_id: Optional[str] = None,
        base_dir: str = "experiments",
    ):
        """
        Initialize the experiment tracker.

        Args:
            experiment_name: Name of the experiment
            run_id: Unique identifier for this run (default: auto-generated)
            base_dir: Base directory for experiment data
        """
        self.experiment_name = experiment_name

        # Generate run ID if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp}"
        else:
            self.run_id = run_id

        # Set up directories
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        self.run_dir = os.path.join(self.experiment_dir, self.run_id)

        self.model_dir = os.path.join(self.run_dir, "models")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.artifacts_dir = os.path.join(self.run_dir, "artifacts")

        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # Set up loggers
        self.logger = get_logger(f"experiment.{experiment_name}.{self.run_id}")
        setup_file_logging(self.logger, self.log_dir, "experiment.log")

        self.json_logger = JsonLogger(self.log_dir, "metrics.jsonl")

        self.logger.info(
            f"Initialized experiment: {experiment_name}, run: {self.run_id}"
        )
        self.logger.info(f"Experiment directory: {self.run_dir}")

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log experiment parameters.

        Args:
            parameters: Dictionary of parameter values
        """
        # Log to both regular and JSON logger
        param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        self.logger.info(f"Parameters: {param_str}")

        self.json_logger.log_parameters(parameters)

        # Also save as JSON file for easy reference
        params_path = os.path.join(self.run_dir, "parameters.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(parameters, f, indent=2)

    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric values
            step: Training step or epoch
            prefix: Prefix for metric names
        """
        # Log to both regular and JSON logger
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step} - {prefix}Metrics: {metric_str}")

        self.json_logger.log_metrics(metrics, step, prefix)

    def save_model(self, model, name: str) -> str:
        """
        Save a model.

        Args:
            model: Model to save
            name: Model name

        Returns:
            Path to the saved model
        """
        model_path = os.path.join(self.model_dir, name)

        # Save model based on type
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(model_path)
        else:
            # Fallback for models without save_pretrained
            os.makedirs(model_path, exist_ok=True)
            torch_path = os.path.join(model_path, "model.pt")
            import torch

            torch.save(model.state_dict(), torch_path)

        self.logger.info(f"Saved model '{name}' to {model_path}")

        # Log as artifact
        self.json_logger.log_artifact(model_path, {"type": "model", "name": name})

        return model_path

    def save_artifact(
        self, data: Any, name: str, artifact_type: str = "general"
    ) -> str:
        """
        Save an artifact.

        Args:
            data: Artifact data
            name: Artifact name
            artifact_type: Type of artifact

        Returns:
            Path to the saved artifact
        """
        # Determine file extension and save method based on type
        if artifact_type == "json":
            ext = ".json"
            save_fn = lambda d, f: json.dump(d, f, indent=2)
        elif artifact_type == "text":
            ext = ".txt"
            save_fn = lambda d, f: f.write(d)
        elif artifact_type == "pickle":
            ext = ".pkl"
            save_fn = lambda d, f: pickle.dump(d, f)
        else:
            # Default to JSON
            ext = ".json"
            save_fn = lambda d, f: json.dump(d, f, indent=2)

        # Add extension if not present
        if not name.endswith(ext):
            name = name + ext

        # Save artifact
        artifact_path = os.path.join(self.artifacts_dir, name)

        with open(
            artifact_path,
            "w" if artifact_type in ("json", "text") else "wb",
            encoding="utf-8" if artifact_type in ("json", "text") else None,
        ) as f:
            save_fn(data, f)

        self.logger.info(f"Saved artifact '{name}' to {artifact_path}")

        # Log as artifact
        self.json_logger.log_artifact(
            artifact_path, {"type": artifact_type, "name": name}
        )

        return artifact_path

    def get_run_info(self) -> Dict[str, Any]:
        """
        Get information about the current run.

        Returns:
            Dictionary of run information
        """
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "model_dir": self.model_dir,
            "log_dir": self.log_dir,
            "artifacts_dir": self.artifacts_dir,
        }


# Set up global logger
logger = get_logger("nlp")


def set_global_log_level(level: Union[int, str]):
    """Set the global log level for the NLP package."""
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)
