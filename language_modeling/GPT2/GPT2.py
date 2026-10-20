import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb

class GPT2FineTuner:
    def __init__(self, model_name="gpt2", dataset_path="dataset.csv", output_dir="./results"):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir

        # Initialiser le tokenizer et le modèle
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        # Ajouter un token de padding si nécessaire
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Charger le dataset avec pandas
        data = pd.read_csv(self.dataset_path)
        print(data.head())  # Afficher les premières lignes du dataset
        
        # Vérifier que la colonne 'text' existe dans le dataset
        if 'text' not in data.columns:
            raise ValueError("Le dataset doit contenir une colonne 'text'.")
        
        self.dataset = Dataset.from_pandas(data)

    def tokenize_function(self, examples):
        # Tokeniser les données et ajouter les labels
        return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=128, return_tensors="pt")

    def preprocess_data(self):
        # Tokeniser le dataset
        self.tokenized_dataset = self.dataset.map(self.tokenize_function, batched=True)

    def data_collator(self):
        # Retourner un DataCollator pour Causal LM
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Ne pas utiliser MLM (Masked Language Modeling), mais Causal LM
        )

    def train(self, learning_rate=5e-5, batch_size=8, epochs=3):
        # Définir le token de padding pour le modèle
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Initialiser W&B pour suivre le fine-tuning
        wandb.init(
            project="gpt2-finetuning",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "architecture": "GPT-2",
                "dataset": "Custom Dataset",
            }
        )

        # Arguments pour l'entraînement
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="no",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            save_strategy="epoch",           # Sauvegarde après chaque epoch
            logging_dir="./logs",
            logging_steps=100,
        )

        # Initialisation du Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=self.data_collator(),
        )

        # Entraîner le modèle
        trainer.train()

        # Sauvegarde du modèle fine-tuné
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Terminer la session W&B
        wandb.finish()

    def generate(self, prompt, max_length=50, num_return_sequences=1):
        from transformers import pipeline

        # Charger le modèle fine-tuné si nécessaire (lors du premier appel)
        fine_tuned_model = GPT2LMHeadModel.from_pretrained(self.output_dir)
        fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(self.output_dir)

        generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
        return generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
