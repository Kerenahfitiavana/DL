import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd

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

        tokens = self.tokenizer(text)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        tokens = tokens + [0] * (self.max_len - len(tokens))

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
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

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output

def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()

    accuracy = correct_predictions / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy
