from src.utils.prepare_data import prepare_and_save_datasets
from src.utils.data_loader import load_dataset_from_folder, load_from_disk
from src.phase1_component_extraction.preprocess import preprocess_component, tokenize_component
from src.phase2_aspect_extraction.preprocess import preprocess_absa, tokenize_absa
from src.phase1_component_extraction.component_trainer import ComponentTrainer
from src.phase2_aspect_extraction.absa_trainer import ABSATrainer
from src.phase1_component_extraction.component_predictor import ComponentPredictor
from src.phase2_aspect_extraction.absa_predictor import ABSAPredictor
from src.utils.evaluator import evaluate_component_outputs, evaluate_absa_outputs
import os

def main():
    # 1. Prepare the dataset by splitting the raw data into train/val/test if not already done
    prepare_and_save_datasets("data/raw/full_dataset.csv", "data/processed")

    # 2. Load and preprocess the component extraction dataset
    comp_path = "data/preprocessed/component"
    if not os.path.exists(comp_path):
        # Load raw processed data from folder
        raw_comp_dataset = load_dataset_from_folder("data/processed")
        # Preprocess data (create input_text and target_text)
        processed_comp_dataset = raw_comp_dataset.map(preprocess_component, batched=True)
        # Tokenize data for model input
        tokenized_comp_dataset = processed_comp_dataset.map(tokenize_component, batched=True)
        # Save preprocessed and tokenized dataset for future reuse
        tokenized_comp_dataset.save_to_disk(comp_path)
    else:
        # Load preprocessed dataset from disk if it exists
        tokenized_comp_dataset = load_from_disk(comp_path)

    # 3. Train the component extraction model using the prepared dataset
    comp_trainer = ComponentTrainer(tokenized_comp_dataset)
    comp_trainer.train()

    # 4. Load and preprocess the ABSA dataset (aspect-based sentiment analysis)
    absa_path = "data/preprocessed/absa"
    if not os.path.exists(absa_path):
        # Load raw processed data from folder
        raw_absa_dataset = load_dataset_from_folder("data/processed")
        # Preprocess ABSA data (create input_text and target_text)
        processed_absa_dataset = raw_absa_dataset.map(preprocess_absa, batched=True)
        # Tokenize ABSA data for model input
        tokenized_absa_dataset = processed_absa_dataset.map(tokenize_absa, batched=True)
        # Save preprocessed and tokenized ABSA dataset for future reuse
        tokenized_absa_dataset.save_to_disk(absa_path)
    else:
        # Load preprocessed ABSA dataset from disk if it exists
        tokenized_absa_dataset = load_from_disk(absa_path)

    # 5. Train the ABSA model using the prepared dataset
    absa_trainer = ABSATrainer(tokenized_absa_dataset)
    absa_trainer.train()

    # 6. Predict and evaluate the component extraction results on the test set
    component_predictor = ComponentPredictor(".models/t5_component_extraction")
    comp_test_inputs = tokenized_comp_dataset["test"]["input_text"]
    comp_test_refs = tokenized_comp_dataset["test"]["target_text"]
    comp_preds = component_predictor.predict(comp_test_inputs)
    evaluate_component_outputs(comp_preds, comp_test_refs)

    # 7. Predict and evaluate the ABSA results on the test set
    absa_predictor = ABSAPredictor(".models/t5_absa")
    absa_test_inputs = tokenized_absa_dataset["test"]["Component sentence"]
    absa_test_refs = tokenized_absa_dataset["test"]["target_text"]
    absa_preds = absa_predictor.predict(absa_test_inputs)
    evaluate_absa_outputs(absa_test_refs, absa_preds)

if __name__ == "__main__":
    main()
