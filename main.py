import argparse
import os

from src.utils.prepare_data import prepare_and_save_datasets
from src.utils.data_loader import load_dataset_from_folder, load_from_disk
from src.phase1_component_extraction.preprocess import preprocess_component, tokenize_component
from src.phase2_aspect_extraction.preprocess import preprocess_absa, tokenize_absa
from src.phase1_component_extraction.component_trainer import ComponentTrainer
from src.phase2_aspect_extraction.absa_trainer import ABSATrainer
from src.phase1_component_extraction.component_predictor import ComponentPredictor
from src.phase2_aspect_extraction.absa_predictor import ABSAPredictor
from src.utils.evaluator import evaluate_component_outputs, evaluate_absa_outputs
from src.pipeline.run_pipeline import run_absa_pipeline

def main(args):
    # Step 1: Prepare dataset (split into train/val/test)
    if args.prepare_data:
        prepare_and_save_datasets(args.raw_data_path, args.save_dir)

    # Step 2: Preprocess component extraction dataset if needed
    if args.train_component or args.eval_component:
        comp_path = "data/preprocessed/component"
        if not os.path.exists(comp_path):
            raw_comp_dataset = load_dataset_from_folder(args.save_dir)
            processed_comp_dataset = raw_comp_dataset.map(preprocess_component, batched=True)
            tokenized_comp_dataset = processed_comp_dataset.map(tokenize_component, batched=True)
            tokenized_comp_dataset.save_to_disk(comp_path)
        else:
            tokenized_comp_dataset = load_from_disk(comp_path)

    # Step 3: Train component model
    if args.train_component:
        comp_trainer = ComponentTrainer(tokenized_comp_dataset)
        comp_trainer.train()

    # Step 4: Evaluate component extraction
    if args.eval_component:
        component_predictor = ComponentPredictor(args.component_model_path)
        comp_test_inputs = tokenized_comp_dataset["test"]["input_text"]
        comp_test_refs = tokenized_comp_dataset["test"]["target_text"]
        comp_preds = component_predictor.predict(comp_test_inputs)
        evaluate_component_outputs(comp_preds, comp_test_refs)

    # Step 5: Preprocess ABSA dataset if needed
    if args.train_absa or args.eval_absa:
        absa_path = "data/preprocessed/absa"
        if not os.path.exists(absa_path):
            raw_absa_dataset = load_dataset_from_folder(args.save_dir)
            processed_absa_dataset = raw_absa_dataset.map(preprocess_absa, batched=True)
            tokenized_absa_dataset = processed_absa_dataset.map(tokenize_absa, batched=True)
            tokenized_absa_dataset.save_to_disk(absa_path)
        else:
            tokenized_absa_dataset = load_from_disk(absa_path)

    # Step 6: Train ABSA model
    if args.train_absa:
        absa_trainer = ABSATrainer(tokenized_absa_dataset)
        absa_trainer.train()

    # Step 7: Evaluate ABSA model
    if args.eval_absa:
        absa_predictor = ABSAPredictor(args.absa_model_path)
        absa_test_inputs = tokenized_absa_dataset["test"]["Component sentence"]
        absa_test_refs = tokenized_absa_dataset["test"]["target_text"]
        absa_preds = absa_predictor.predict(absa_test_inputs)
        evaluate_absa_outputs(absa_test_refs, absa_preds)

    # Step 8: Run full ABSA pipeline on real reviews and export results
    if args.run_pipeline:
        run_absa_pipeline(
            input_csv=args.pipeline_input,
            output_csv=args.pipeline_output,
            component_model_path=args.component_model_path,
            absa_model_path=args.absa_model_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA Pipeline Runner")

    # Flags to control workflow
    parser.add_argument("--prepare_data", action="store_true", help="Prepare and split dataset")
    parser.add_argument("--train_component", action="store_true", help="Train component extraction model")
    parser.add_argument("--eval_component", action="store_true", help="Evaluate component extraction model")
    parser.add_argument("--train_absa", action="store_true", help="Train ABSA model")
    parser.add_argument("--eval_absa", action="store_true", help="Evaluate ABSA model")
    parser.add_argument("--run_pipeline", action="store_true", help="Run full ABSA pipeline")

    # Paths
    parser.add_argument("--raw_data_path", type=str, default="data/raw/full_dataset.csv")
    parser.add_argument("--save_dir", type=str, default="data/processed")
    parser.add_argument("--component_model_path", type=str, default=".models/t5_component_extraction")
    parser.add_argument("--absa_model_path", type=str, default=".models/t5_absa")
    parser.add_argument("--pipeline_input", type=str, default="data/raw/full_reviews.csv")
    parser.add_argument("--pipeline_output", type=str, default="pipeline/output.csv")

    args = parser.parse_args()
    main(args)
