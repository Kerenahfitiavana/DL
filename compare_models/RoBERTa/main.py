import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_classification_pipeline import create_data_loaders, train_epoch, evaluate

DATA_PATH = "../spam.csv"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    
    train_loader, test_loader = create_data_loaders(DATA_PATH, tokenizer, MAX_LEN, BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}, F1 Score: {train_f1:.4f}")
        
        val_loss, accuracy, val_f1 = evaluate(model, test_loader, device)
        print(f"Validation loss: {val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, F1 Score: {val_f1:.4f}")

if __name__ == '__main__':
    main()
