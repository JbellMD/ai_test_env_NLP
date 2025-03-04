# NLP Test Environment

This environment provides tools for training, evaluating, and deploying NLP models.

## Features
- Text classification using transformer models
- Model training and evaluation
- REST API deployment

## Setup
1. Build the Docker image:
```bash
docker build -t nlp_env .

## Run the Docker container:
docker run -it -p 8888:8888 -p 8000:8000 nlp_env

## Test the REST API
```bash
curl http://localhost:8888/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test."}'
```

## Deploy the model
```bash
docker run -it -p 8000:8000 nlp_env
```
## Training the model
```bash
docker run -it -p 8888:8888 -p 8000:8000 nlp_env
python scripts/train.py --dataset imdb --model bert-base-uncased --epochs 3 --batch_size 16
```

## Evaluate the model
```bash
docker run -it -p 8888:8888 -p 8000:8000 nlp_env
python scripts/evaluate.py --dataset imdb --model model.pth
```
## Deploy the model
```bash
docker run -it -p 8000:8000 nlp_env
python scripts/deploy.py --model ./models/bert-base-uncased
```