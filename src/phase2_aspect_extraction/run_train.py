from utils.data_loader import load_dataset_from_folder
from preprocess import preprocess_absa, tokenize_absa
from datasets import load_from_disk
from absa_trainer import ABSATrainer
import os

def main():
    # Path to save/load the preprocessed ABSA dataset
    preprocessed_path = "data/preprocessed/component"

    # If the preprocessed/tokenized dataset already exists, load it directly
    if os.path.exists(preprocessed_path):
        dataset = load_from_disk(preprocessed_path)
        print("Loaded preprocessed dataset from disk.")
    else:
        # Otherwise, load the raw dataset (CSV files that were already split)
        raw_dataset = load_dataset_from_folder("data/processed")

        # Preprocess: generate input_text and target_text from raw columns
        processed_dataset = raw_dataset.map(preprocess_absa, batched=True)

        # Tokenize: convert text into model inputs (input_ids, attention_mask, labels, etc.)
        tokenized_dataset = processed_dataset.map(tokenize_absa, batched=True)

        # Save the processed dataset for future reuse
        tokenized_dataset.save_to_disk(preprocessed_path)
        print("Preprocessed and saved dataset to disk.")

        dataset = tokenized_dataset

    # Initialize trainer and start training
    trainer = ABSATrainer(tokenizer=None, dataset=dataset)  # tokenizer=None if built into trainer
    trainer.train()

# Run the training pipeline if this file is executed directly
if __name__ == "__main__":
    main()
