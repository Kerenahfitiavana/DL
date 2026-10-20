import torch
from torch import nn
from LSTM import create_data_loaders, LSTMClassifier, train_epoch, evaluate

DATA_PATH = "../spam.csv"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 1e-3
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 2
VOCAB_SIZE = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenizer(text):
    return [ord(char) for char in text.lower() if ord(char) < 128]

def main():
    train_loader, test_loader = create_data_loaders(
        DATA_PATH, tokenizer, MAX_LEN, BATCH_SIZE
    )

    model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train loss: {train_loss}")

        val_loss, val_accuracy = evaluate(model, test_loader, loss_fn, device)
        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_accuracy}")

    torch.save(model.state_dict(), "./lstm_spam_classifier.pth")

if __name__ == "__main__":
    main()