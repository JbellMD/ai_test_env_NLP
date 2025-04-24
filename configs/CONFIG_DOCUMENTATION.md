# Configuration Documentation

This document provides detailed explanations for the configuration files used in the NLP Toolkit.

## API Configuration (`api_config.json`)

The API configuration controls the behavior of the FastAPI server and how models are loaded and accessed.

```json
{
  "api_version": "1.0.0",       // Version number for the API
  "host": "0.0.0.0",            // Host address to bind the server to (0.0.0.0 allows external connections)
  "port": 8000,                 // Port to run the API server on
  "debug": false,               // Enable/disable debug mode
  "log_level": "INFO",          // Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  
  "cors": {                     // Cross-Origin Resource Sharing settings
    "allow_origins": ["*"],     // Origins allowed to access the API ("*" = all)
    "allow_methods": ["*"],     // HTTP methods allowed (GET, POST, etc.)
    "allow_headers": ["*"]      // HTTP headers allowed in requests
  },
  
  "rate_limiting": {            // Rate limiting to prevent abuse
    "enabled": true,            // Enable/disable rate limiting
    "requests_per_minute": 60   // Maximum requests allowed per minute per client
  },
  
  "default_models": {           // Default models to use for each task
    "classification": "default_classification",
    "ner": "default_ner",
    "sentiment": "default_sentiment",
    "summarization": "default_summarization"
  },
  
  "model_loading": {            // Model loading behavior
    "preload_models": [         // Models to load on server startup
      "default_classification", 
      "default_ner", 
      "default_sentiment"
    ],
    "lazy_load": true,          // Load models on-demand vs. all at startup
    "cache_size": 5             // Number of models to keep in memory
  },
  
  "security": {                 // Security settings
    "require_api_key": false,   // Require API key for authentication
    "enable_ssl": false         // Enable SSL/TLS encryption
  }
}
```

## Model Configurations (`model_configs/*.json`)

Each model configuration file defines parameters for a specific model architecture. Below is the documentation for the BERT configuration:

### BERT Configuration (`model_configs/bert_config.json`)

```json
{
  "model_name": "bert-base-uncased",  // Hugging Face model identifier
  "task": "classification",           // NLP task this config is optimized for
  "num_labels": 2,                    // Number of output classes/labels
  "max_length": 512,                  // Maximum sequence length
  "batch_size": 16,                   // Default batch size for training/inference
  
  "training": {                       // Training parameters
    "learning_rate": 2e-5,            // Default learning rate
    "weight_decay": 0.01,             // L2 regularization strength
    "adam_epsilon": 1e-8,             // Epsilon for Adam optimizer
    "warmup_steps": 0,                // Linear warmup steps
    "gradient_accumulation_steps": 1, // Gradient accumulation for larger batches
    "max_grad_norm": 1.0              // Gradient clipping norm
  },
  
  "preprocessing": {                  // Text preprocessing options
    "lowercase": true,                // Convert text to lowercase
    "remove_special_chars": false,    // Remove special characters
    "normalize_whitespace": true      // Normalize whitespace
  }
}
```

### RoBERTa Configuration (`model_configs/roberta_config.json`)

Similar to BERT but with RoBERTa-specific settings.

### DistilBERT Configuration (`model_configs/distilbert_config.json`)

Lighter version of BERT with similar parameters but optimized for efficiency.

## Modifying Configurations

When modifying configurations, consider the following:

1. **API Configuration**:
   - Change `host` and `port` to match your deployment environment
   - Adjust `cors` settings for production use (limit to specific origins)
   - Enable `require_api_key` for secure deployments
   - Tune `cache_size` based on available memory

2. **Model Configurations**:
   - Fine-tune `batch_size` based on available GPU memory
   - Adjust `learning_rate` and other training parameters for specific datasets
   - Modify `preprocessing` options based on your text data characteristics

Always restart the server after changing configurations to ensure changes take effect.
