import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader

def evaluate_model(dataset_name, model_path):
    # Load dataset
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Prepare DataLoader
    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=16)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())
        true_labels.extend(batch['labels'].tolist())
    
    # Print classification report
    print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name from Hugging Face")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    args = parser.parse_args()
    
    evaluate_model(args.dataset, args.model)