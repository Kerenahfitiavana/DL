import pandas as pd
from LSTM.LSTM import EmbeddingLSTM
from BERT.BERT import BERTClassifier

DATA_PATH = "../spam.csv"

def compare_models():
    try:
        # Chargement des données
        df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1', usecols=["v1", "v2"])

        print("\n==== Évaluation du modèle LSTM ====")
        # Paramètres pour LSTM
        lstm_vocab_size = 5000
        lstm_embedding_dim = 64
        lstm_max_length = 200
        lstm_model = EmbeddingLSTM(lstm_vocab_size, lstm_embedding_dim, lstm_max_length)
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = lstm_model.preprocess_data(df)
        lstm_model.train(X_train_lstm, y_train_lstm, epochs=10, batch_size=32)
        lstm_model.evaluate(X_test_lstm, y_test_lstm)

        print("\n==== Évaluation du modèle BERT ====")
        # Paramètres pour BERT
        bert_max_length = 128
        bert_model = BERTClassifier(max_length=bert_max_length)
        (X_train_bert, masks_train_bert, y_train_bert), (X_test_bert, masks_test_bert, y_test_bert) = bert_model.preprocess_data(df)
        bert_model.train(X_train_bert, masks_train_bert, y_train_bert, epochs=3, batch_size=16)
        bert_model.evaluate(X_test_bert, masks_test_bert, y_test_bert)

    except FileNotFoundError:
        print(f"Erreur : le fichier {DATA_PATH} est introuvable.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    compare_models()
