import torch
from transformers import BertTokenizer, BertForSequenceClassification
from BERT import create_data_loaders, train_epoch, evaluate

DATA_PATH = "../spam.csv"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader, test_loader = create_data_loaders(
        DATA_PATH, tokenizer, MAX_LEN, BATCH_SIZE
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss}")

        val_loss, val_accuracy = evaluate(model, test_loader, device)
        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_accuracy}")

    model.save_pretrained("./bert_spam_classifier")
    tokenizer.save_pretrained("./bert_spam_classifier")

if __name__ == "__main__":
    main()
