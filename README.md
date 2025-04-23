# NLP Toolkit

A comprehensive, modular, and extensible NLP toolkit supporting multiple tasks, models, and deployment options, designed for both research and production environments.

## Features

- **Multi-Task Support**: Classification, Named Entity Recognition (NER), Sentiment Analysis, and Summarization (both extractive and abstractive)
- **Model Hub**: Support for various transformer architectures (BERT, RoBERTa, DistilBERT, T5, etc.)
- **Advanced Training**: Parameter-efficient fine-tuning, few-shot learning, and ensemble models
- **Comprehensive Evaluation**: Task-specific metrics, visualizations, and detailed reporting
- **API-First Design**: FastAPI-based RESTful API with model registry and task-specific endpoints
- **Extensible Architecture**: Easily add new tasks, models, and customizations

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-username/nlp_toolkit.git
cd nlp_toolkit

# Install the package in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,test,docs]"
```

### Using Docker

```bash
# Build the Docker image
docker build -t nlp_toolkit .

# Run the container
docker run -p 8000:8000 -p 8888:8888 -v $(pwd):/app nlp_toolkit
```

## Project Structure

```
nlp_toolkit/
├── configs/                # Configuration files
│   ├── api_config.json     # API configuration
│   └── model_configs/      # Model-specific configurations
├── notebooks/              # Jupyter notebooks for demos and exploration
├── scripts/                # Command-line scripts
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── deploy.py           # Deployment script
├── src/                    # Source code
│   ├── api/                # API modules
│   ├── data/               # Data processing modules
│   ├── models/             # Model definitions
│   ├── training/           # Training utilities
│   └── utils/              # Utility functions
├── tests/                  # Unit and integration tests
├── Dockerfile              # Docker configuration
├── requirements.txt        # Dependencies
├── setup.py                # Package installation
└── README.md               # This file
```

## Usage

### Command-Line Interface

#### Training

```bash
# Basic usage
python scripts/train.py --task classification --dataset imdb --model bert-base-uncased

# Advanced options
python scripts/train.py \
  --task classification \
  --dataset imdb \
  --model bert-base-uncased \
  --output_dir ./models/my_classifier \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --log_level info \
  --track_metrics \
  --visualize
```

#### Evaluation

```bash
# Basic usage
python scripts/evaluate.py --task classification --model_path ./models/my_classifier --dataset imdb

# With visualization
python scripts/evaluate.py \
  --task classification \
  --model_path ./models/my_classifier \
  --dataset imdb \
  --output_dir ./results/my_evaluation \
  --visualize \
  --detailed_report
```

#### Deployment

```bash
# Start the API server
python scripts/deploy.py --model_dir ./models --config ./configs/api_config.json

# With security options
python scripts/deploy.py \
  --model_dir ./models \
  --host 0.0.0.0 \
  --port 8000 \
  --enable_cors \
  --cors_origins "http://localhost:3000,https://example.com"
```

### Python API

```python
from src.models.classifier import TransformerClassifier
from src.models.named_entity_recognition import NERModel
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.summarizer import TextSummarizer, ExtractiveSummarizer

# Text Classification
classifier = TransformerClassifier("bert-base-uncased", num_labels=2)
classifier.train(train_dataloader, val_dataloader, epochs=3)
predictions = classifier.predict(["This is a positive review", "This is terrible"])

# Named Entity Recognition
ner_model = NERModel("dslim/bert-base-NER")
entities = ner_model.predict("Apple Inc. is headquartered in Cupertino, California.")

# Sentiment Analysis
sentiment_analyzer = SentimentAnalyzer("distilbert-base-uncased-finetuned-sst-2-english")
sentiments = sentiment_analyzer.analyze_sentiment(["I love this product", "The service was poor"])

# Summarization
summarizer = TextSummarizer("facebook/bart-large-cnn")
summary = summarizer.summarize_text("Long article text here...", max_length=100)

# Extractive Summarization
ext_summarizer = ExtractiveSummarizer(method="textrank")
summary = ext_summarizer.summarize("Long article text here...", ratio=0.3)
```

### Web API

Once deployed, the API provides the following endpoints:

- `/api/health`: Health check endpoint
- `/api/v1/classification/predict`: Text classification prediction
- `/api/v1/ner/predict`: Named entity recognition prediction
- `/api/v1/sentiment/analyze`: Sentiment analysis prediction
- `/api/v1/summarization/summarize`: Text summarization
- `/api/v1/models`: List available models
- `/api/v1/models/{model_id}`: Get model details and metadata

Example API request:

```bash
curl -X POST "http://localhost:8000/api/v1/classification/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an example text", "model_id": "bert-base-uncased"}'
```

## Development

### Adding a New Task

1. Create a new model module in `src/models/`
2. Implement the required interfaces (training, evaluation, prediction)
3. Add task-specific metrics in `src/training/metrics.py`
4. Add API endpoints in `src/api/routes.py`
5. Update the model registry in `src/api/model_registry.py`

### Adding a New Model

1. Create a model configuration in `configs/model_configs/`
2. Implement or extend the appropriate model class
3. Register the model in the model registry

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers
- FastAPI
- PyTorch
- scikit-learn
- And all other libraries used in this project