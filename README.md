# NLP Toolkit for ReadyTensor

A comprehensive, modular, and extensible NLP toolkit supporting multiple tasks, models, and deployment options, designed for both research and production environments. This project provides a complete solution for building, training, evaluating, and deploying state-of-the-art NLP models across various tasks and domains.

## Features

- **Multi-Task Support**: Classification, Named Entity Recognition (NER), Sentiment Analysis, and Summarization (both extractive and abstractive)
- **Model Hub**: Support for various transformer architectures (BERT, RoBERTa, DistilBERT, T5, etc.)
- **Advanced Training**: Parameter-efficient fine-tuning, few-shot learning, and ensemble models
- **Comprehensive Evaluation**: Task-specific metrics, visualizations, and detailed reporting
- **API-First Design**: FastAPI-based RESTful API with model registry and task-specific endpoints
- **Extensive Testing**: Unit tests for all components ensuring reliability and quality
- **Demo Notebooks**: Interactive examples showcasing capabilities and use cases
- **Benchmark Results**: Performance metrics across models and tasks for informed decision-making
- **Extensible Architecture**: Easily add new tasks, models, and customizations

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/JbellMD/ai_test_env_NLP.git
cd ai_test_env_NLP

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
ai_test_env_NLP/
├── configs/                # Configuration files
│   ├── api_config.json     # API configuration
│   └── model_configs/      # Model-specific configurations
├── notebooks/              # Jupyter notebooks for demos and benchmarks
│   ├── 01_text_classification_demo.ipynb  # Classification examples
│   ├── 02_named_entity_recognition_demo.ipynb  # NER examples
│   ├── 03_text_summarization_demo.ipynb  # Summarization examples
│   └── 04_benchmarking_results.ipynb  # Performance benchmarks
├── scripts/                # Command-line scripts
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── deploy.py           # Deployment script
├── src/                    # Source code
│   ├── api/                # API modules
│   │   ├── app.py          # FastAPI application
│   │   ├── routes.py       # API endpoints
│   │   └── model_registry.py  # Model management
│   ├── data/               # Data processing modules
│   │   ├── data_loader.py  # Dataset loading
│   │   └── preprocessing.py  # Text preprocessing
│   ├── models/             # Model definitions
│   │   ├── classifier.py   # Classification models
│   │   ├── named_entity_recognition.py  # NER models
│   │   ├── sentiment_analyzer.py  # Sentiment models
│   │   └── summarizer.py   # Summarization models
│   ├── training/           # Training utilities
│   │   ├── trainer.py      # Training loops
│   │   └── metrics.py      # Evaluation metrics
│   └── utils/              # Utility functions
│       ├── logging_utils.py  # Logging utilities
│       └── visualization.py  # Visualization tools
├── tests/                  # Unit and integration tests
│   ├── test_api/           # API tests
│   ├── test_data/          # Data processing tests
│   ├── test_models/        # Model tests
│   └── test_training/      # Training tests
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

## Testing

The project includes comprehensive unit tests for all components:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_models/test_classifier.py

# Run with coverage report
pytest --cov=src tests/
```

## Demo Notebooks

The `notebooks/` directory contains interactive examples demonstrating the toolkit's capabilities:

1. **Text Classification Demo** (`01_text_classification_demo.ipynb`):
   - Loading and preprocessing data for classification
   - Training a transformer-based classifier
   - Evaluating model performance with visualizations
   - Making predictions on new data

2. **Named Entity Recognition Demo** (`02_named_entity_recognition_demo.ipynb`):
   - Using pre-trained NER models
   - Fine-tuning on custom datasets
   - Visualizing entity predictions
   - Real-world text entity extraction

3. **Text Summarization Demo** (`03_text_summarization_demo.ipynb`):
   - Abstractive and extractive summarization techniques
   - Comparing different summarization methods
   - Evaluating summaries with ROUGE metrics
   - Real-world document summarization

4. **Benchmarking Results** (`04_benchmarking_results.ipynb`):
   - Performance metrics across all tasks and models
   - Comparative analysis of model architectures
   - Speed/accuracy trade-offs
   - Visualization of benchmark results

## Development

### Adding a New Task

1. Create a new model module in `src/models/`
2. Implement the required interfaces (training, evaluation, prediction)
3. Add task-specific metrics in `src/training/metrics.py`
4. Add API endpoints in `src/api/routes.py`
5. Update the model registry in `src/api/model_registry.py`
6. Add tests in the appropriate `tests/` subdirectory

### Adding a New Model

1. Create a model configuration in `configs/model_configs/`
2. Implement or extend the appropriate model class
3. Register the model in the model registry
4. Add benchmarking results in the benchmark notebook

## ReadyTensor Submission

This project has been structured to meet the requirements for ReadyTensor NLP submission:

1. **Technical Rigor**: Comprehensive implementation of multiple NLP tasks
2. **Originality**: Advanced architectures and techniques beyond basic examples
3. **Clarity**: Well-documented code with clear structure and API design
4. **Practical Impact**: Ready-to-use for real-world applications with benchmarks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers
- FastAPI
- PyTorch
- scikit-learn
- sumy, rouge, nltk
- And all other libraries used in this project