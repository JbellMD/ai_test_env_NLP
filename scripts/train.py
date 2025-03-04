import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(dataset_name, model_name, num_epochs=3, batch_size=16):
    # Load dataset
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Prepare DataLoader
    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=batch_size)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(set(dataset['train']['label']))
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        predictions, true_labels = [], []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
            true_labels.extend(batch['labels'].tolist())
        
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch + 1} - Loss: {total_loss/len(train_dataloader):.4f}, Accuracy: {accuracy:.4f}")
    
    # Save model
    model.save_pretrained(f"./models/{model_name.replace('/', '_')}")
    tokenizer.save_pretrained(f"./models/{model_name.replace('/', '_')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name from Hugging Face")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name from Hugging Face")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    train_model(args.dataset, args.model, args.epochs, args.batch_size)