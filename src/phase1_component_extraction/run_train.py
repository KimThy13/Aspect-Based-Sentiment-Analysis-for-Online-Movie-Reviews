from utils.data_loader import load_dataset_from_folder
from preprocess import preprocess_component, tokenize_component
from datasets import load_from_disk
from component_trainer import ComponentTrainer
import os

def main():
    preprocessed_path = "data/preprocessed/component"

    # If the dataset has already been preprocessed and tokenized, load it from disk
    if os.path.exists(preprocessed_path):
        dataset = load_from_disk(preprocessed_path)
        print("✅ Loaded preprocessed dataset from disk.")
    else:
        # Load the raw dataset (should already be split into train/validation/test)
        raw_dataset = load_dataset_from_folder("data/processed")

        # Step 1: Preprocess to create input_text and target_text
        processed_dataset = raw_dataset.map(preprocess_component, batched=True)

        # Step 2: Tokenize the input and target text into input_ids and labels
        tokenized_dataset = processed_dataset.map(tokenize_component, batched=True)

        # Save the tokenized dataset to disk for future use
        tokenized_dataset.save_to_disk(preprocessed_path)
        print("✅ Preprocessed and saved dataset to disk.")

        dataset = tokenized_dataset

    # Initialize the trainer with the dataset and start training the model
    trainer = ComponentTrainer(dataset)
    trainer.train()

# Entry point of the script
if __name__ == "__main__":
    main()
