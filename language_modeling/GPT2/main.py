from GPT2 import GPT2FineTuner

def main():
    # Initialiser l'instance de GPT2FineTuner avec les paramètres nécessaires
    fine_tuner = GPT2FineTuner(
        model_name="gpt2",
        dataset_path="/content/drive/My Drive/dataset/dataset.csv",
        output_dir="./results"
    )

    # Prétraitement des données
    print("Prétraitement des données...")
    fine_tuner.preprocess_data()

    # Entraîner le modèle
    print("Début de l'entraînement...")     
    fine_tuner.train(learning_rate=5e-5, batch_size=8, epochs=3)

if __name__ == "__main__":
    main()




# '/content/drive/My Drive/dataset/dataset.csv'
