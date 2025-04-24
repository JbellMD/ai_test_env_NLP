# NLP Toolkit API Documentation

This document describes the REST API endpoints provided by the NLP Toolkit.

## Base URL

All API endpoints are relative to the base URL: `http://hostname:port/api/`

## Authentication

Authentication is not required for development but should be implemented for production deployments.

## Endpoints Overview

The API is organized around the following NLP tasks:

1. **Classification** - Text classification into predefined categories
2. **Named Entity Recognition (NER)** - Identifying named entities in text
3. **Sentiment Analysis** - Analyzing sentiment in text
4. **Summarization** - Generating summaries of longer texts
5. **Model Hub** - Listing and getting information about available models

---

## Classification API

### Classify Text

Classify text into predefined categories.

**Endpoint:** `POST /api/classification/classify`

**Request Body:**

```json
{
  "text": "string or array of strings",
  "model_name": "string (optional)",
  "return_probabilities": false
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string or array of strings | Yes | Text to classify |
| `model_name` | string | No | Model name to use for classification. If not provided, a default model will be used |
| `return_probabilities` | boolean | No | Whether to return class probabilities (default: false) |

**Response:**

```json
{
  "predictions": ["class1", "class2"],
  "probabilities": [
    {"class1": 0.8, "class2": 0.2},
    {"class1": 0.3, "class2": 0.7}
  ],
  "model_name": "bert-base-uncased-sst2"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | array of strings/integers | Predicted classes for each input text |
| `probabilities` | array of objects | Class probabilities for each input text (only if requested) |
| `model_name` | string | Model used for prediction |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/classification/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie is fantastic", "return_probabilities": true}'
```

---

## Named Entity Recognition (NER) API

### Recognize Entities

Identify named entities in text.

**Endpoint:** `POST /api/ner/recognize`

**Request Body:**

```json
{
  "text": "string or array of strings",
  "model_name": "string (optional)",
  "merge_entities": false
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string or array of strings | Yes | Text for entity recognition |
| `model_name` | string | No | Model name to use for NER. If not provided, a default model will be used |
| `merge_entities` | boolean | No | Whether to merge adjacent entities of the same type (default: false) |

**Response:**

```json
{
  "entities": [
    [
      {
        "entity": "Microsoft",
        "label": "ORG",
        "start": 0,
        "end": 9
      },
      {
        "entity": "Seattle",
        "label": "LOC",
        "start": 35,
        "end": 42
      }
    ]
  ],
  "model_name": "spacy-en_core_web_sm"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `entities` | array of arrays | For each input text, an array of entity objects |
| `model_name` | string | Model used for prediction |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/ner/recognize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Microsoft was founded in Seattle"}'
```

---

## Sentiment Analysis API

### Analyze Sentiment

Analyze the sentiment expressed in text.

**Endpoint:** `POST /api/sentiment/analyze`

**Request Body:**

```json
{
  "text": "string or array of strings",
  "model_name": "string (optional)",
  "return_probabilities": false,
  "aspects": ["price", "quality"] (optional)
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string or array of strings | Yes | Text for sentiment analysis |
| `model_name` | string | No | Model name to use for sentiment analysis. If not provided, a default model will be used |
| `return_probabilities` | boolean | No | Whether to return sentiment probabilities (default: false) |
| `aspects` | array of strings | No | Specific aspects to analyze (for aspect-based sentiment analysis) |

**Response:**

```json
{
  "sentiments": ["positive", "negative"],
  "probabilities": [
    {"positive": 0.9, "negative": 0.1},
    {"positive": 0.2, "negative": 0.8}
  ],
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `sentiments` | array of strings or objects | Sentiment labels or aspect-sentiment mappings |
| `probabilities` | array of objects | Sentiment probabilities for each input text (only if requested) |
| `model_name` | string | Model used for prediction |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/sentiment/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!", "return_probabilities": true}'
```

**Example Request with Aspects:**

```bash
curl -X POST "http://localhost:8000/api/sentiment/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "The phone has a great camera but the battery life is terrible.", "aspects": ["camera", "battery"]}'
```

---

## Summarization API

### Summarize Text

Generate a summary of longer text.

**Endpoint:** `POST /api/summarization/summarize`

**Request Body:**

```json
{
  "text": "string or array of strings",
  "model_name": "string (optional)",
  "max_length": 128,
  "min_length": 30,
  "method": "abstractive",
  "length": "medium",
  "focus": "string (optional)"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string or array of strings | Yes | Text to summarize |
| `model_name` | string | No | Model name to use for summarization. If not provided, a default model will be used |
| `max_length` | integer | No | Maximum length of the summary in tokens (default: 128) |
| `min_length` | integer | No | Minimum length of the summary in tokens (default: 30) |
| `method` | string | No | Summarization method: "abstractive" or "extractive" (default: "abstractive") |
| `length` | string | No | Summary length preset: "short", "medium", or "long" (default: "medium") |
| `focus` | string | No | Topic to focus on in the summary (optional) |

**Response:**

```json
{
  "summaries": ["This is a summary of the first text", "This is a summary of the second text"],
  "model_name": "facebook/bart-large-cnn"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `summaries` | array of strings | Generated summaries for each input text |
| `model_name` | string | Model used for prediction |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/summarization/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.", "method": "extractive", "length": "short"}'
```

---

## Model Hub API

### List Models

List available models, optionally filtered by task.

**Endpoint:** `GET /api/models/list`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | No | Filter models by task type (e.g., "classification", "ner", "sentiment", "summarization") |

**Response:**

```json
{
  "models": [
    {
      "name": "bert-base-uncased-sst2",
      "task": "classification",
      "description": "BERT base model fine-tuned on SST-2 dataset",
      "size": "base",
      "languages": ["english"],
      "metrics": {"accuracy": 0.92}
    },
    {
      "name": "spacy-en_core_web_sm",
      "task": "ner",
      "description": "SpaCy English small model",
      "size": "small",
      "languages": ["english"],
      "metrics": {"f1": 0.84}
    }
  ]
}
```

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/models/list?task=classification"
```

### Get Model Info

Get detailed information about a specific model.

**Endpoint:** `GET /api/models/{model_name}`

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | Yes | Name of the model to get information about |

**Response:**

```json
{
  "name": "bert-base-uncased-sst2",
  "task": "classification",
  "description": "BERT base model fine-tuned on SST-2 dataset",
  "size": "base",
  "languages": ["english"],
  "metrics": {"accuracy": 0.92},
  "parameters": 110000000,
  "training_data": "SST-2",
  "license": "MIT"
}
```

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/models/bert-base-uncased-sst2"
```

---

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of a request.

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input parameters |
| 404 | Not Found - Model or resource not found |
| 422 | Unprocessable Entity - Request validation error |
| 500 | Internal Server Error |

Error responses include a message explaining the error:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Rate limiting may be enforced in production to ensure fair usage of the API. Please refer to production documentation for details on rate limits.

## Versioning

API versioning is managed through the URL. The current version is v1, which is implied in the base URL.
