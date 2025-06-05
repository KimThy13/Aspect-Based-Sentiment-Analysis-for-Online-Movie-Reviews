from utils.data_loader import load_dataset_from_folder
from preprocess import preprocess_component, tokenize_component
from datasets import load_from_disk
from component_trainer import ComponentTrainer
import os

def main():
    preprocessed_path = "data/preprocessed/component"

    # Nếu dataset đã preprocess/tokenize, thì load trực tiếp
    if os.path.exists(preprocessed_path):
        dataset = load_from_disk(preprocessed_path)
        print("✅ Loaded preprocessed dataset from disk.")
    else:
        # Load raw dataset từ CSV đã split
        raw_dataset = load_dataset_from_folder("data/processed")

        # Preprocess (tạo input_text và target_text)
        processed_dataset = raw_dataset.map(preprocess_component, batched=True)

        # Tokenize (chuyển sang input_ids, labels,...)
        tokenized_dataset = processed_dataset.map(tokenize_component, batched=True)

        # Save để tái sử dụng lần sau
        tokenized_dataset.save_to_disk(preprocessed_path)
        print("✅ Preprocessed and saved dataset to disk.")

        dataset = tokenized_dataset

    # Huấn luyện model
    trainer = ComponentTrainer(dataset)
    trainer.train()

if __name__ == "__main__":
    main()
