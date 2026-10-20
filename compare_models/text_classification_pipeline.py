import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(data_path, tokenizer, max_len, batch_size):
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    df = df.rename(columns={"v1": "label", "v2": "text"})[["label", "text"]]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    texts = df['text'].values
    labels = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = SpamDataset(X_train, y_train, tokenizer, max_len)
    test_dataset = SpamDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(data_loader), f1

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            correct_predictions += torch.sum(preds == labels).item()

    accuracy = correct_predictions / len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(data_loader), accuracy, f1
