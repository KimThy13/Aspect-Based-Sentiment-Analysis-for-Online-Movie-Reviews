import argparse
from functools import partial
import os
import pandas as pd
from datasets import load_from_disk
from pathlib import Path

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)

from src.utils.prepare_data import prepare_and_save_datasets
from src.utils.data_loader import load_dataset_from_folder
from src.phase1_component_extraction.preprocess import preprocess_component, tokenize_component
from src.phase2_aspect_extraction.preprocess import preprocess_absa, tokenize_absa
from src.utils.predictor import Predictor
from src.utils.evaluator import evaluate_component_outputs, evaluate_absa_outputs, evaluate_component_outputs_by_sentence, evaluate_absa_detailed
from src.pipeline.run_pipeline import run_absa_pipeline
from src.utils.trainer import Seq2SeqTrainerWrapper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model_and_tokenizer(model_path_or_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

    # Try to load from safetensors if available
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_name, use_safetensors=True)
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_name)  # fallback

    return tokenizer, model


def detect_model_type(model_path_or_name):
    name = str(model_path_or_name).lower()
    if "t5" in name:
        return "t5"
    elif "bart" in name:
        return "bart"
    else:
        raise ValueError(f"Cannot detect model type from path: {model_path_or_name}")



def main(args):
    # Detect model types
    component_model_type = detect_model_type(args.component_model_path)
    absa_model_type = detect_model_type(args.absa_model_path)

    # Step 1: Prepare dataset (split into train/val/test)
    if args.prepare_data:
        prepare_and_save_datasets(args.raw_data_path, args.save_dir)

    # Step 2: Preprocess component extraction dataset if needed
    if args.train_component or args.eval_component:
        comp_path = "data/preprocessed/component"

        tokenizer, model = load_model_and_tokenizer(str(args.component_model_path))

        if not os.path.exists(comp_path):
            raw_comp_dataset = load_dataset_from_folder(args.save_dir)

            processed_comp_dataset = raw_comp_dataset.map(
                partial(preprocess_component, model_type=component_model_type),
                batched=True,
                remove_columns=raw_comp_dataset["train"].column_names
            )

            tokenized_comp_dataset = processed_comp_dataset.map(
                partial(tokenize_component, tokenizer=tokenizer),
                batched=True
            )

            tokenized_comp_dataset.save_to_disk(comp_path)
        else:
            tokenized_comp_dataset = load_from_disk(comp_path)

    # Step 3: Train component model
    if args.train_component:
        absa_trainer = Seq2SeqTrainerWrapper(
            model=model,
            tokenizer=tokenizer,
            dataset=tokenized_comp_dataset,
            **vars(args) 
        )
        absa_trainer.train()

    # Step 4: Evaluate component extraction
    if args.eval_component:
        component_predictor = Predictor(
            args.component_model_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_beams=args.num_beams
        )
        comp_test_inputs = tokenized_comp_dataset["test"]
        comp_test_refs = tokenized_comp_dataset["test"]["target_text"]
        comp_preds = component_predictor.predict(comp_test_inputs)

        comp_preds = [" ; ".join(p) if isinstance(p, list) else p for p in comp_preds]
        comp_test_refs = [" ; ".join(r) if isinstance(r, list) else r for r in comp_test_refs]

        os.makedirs(args.save_dir, exist_ok=True)
        df = pd.DataFrame({
            "input": tokenized_comp_dataset["test"]["input_text"],
            "reference": comp_test_refs,
            "prediction": comp_preds
        })
        df.to_csv(os.path.join(args.save_dir, "component_extraction_predictions.csv"), index=False)

        component_eval_path = os.path.join(args.save_dir, "component_eval_results.json")
        evaluate_component_outputs(comp_preds, comp_test_refs, output_file=component_eval_path)
        component_eval_path_sentence = os.path.join(args.save_dir, "component_eval_sentence_level.json")
        evaluate_component_outputs_by_sentence(comp_preds, comp_test_refs, output_file=component_eval_path_sentence)

    # Step 5: Preprocess ABSA dataset if needed    
    if args.train_absa or args.eval_absa:
        absa_path = "data/preprocessed/absa"

        tokenizer, model = load_model_and_tokenizer(str(args.absa_model_path))

        if not os.path.exists(absa_path):
            raw_absa_dataset = load_dataset_from_folder(args.save_dir)

            processed_absa_dataset = raw_absa_dataset.map(
                partial(preprocess_absa, model_type=absa_model_type),
                batched=True,
                remove_columns=raw_absa_dataset["train"].column_names
            )

            tokenized_absa_dataset = processed_absa_dataset.map(
                partial(tokenize_absa, tokenizer=tokenizer),
                batched=True
            )

            tokenized_absa_dataset.save_to_disk(absa_path)
        else:
            tokenized_absa_dataset = load_from_disk(absa_path)

    # Step 6: Train ABSA model
    if args.train_absa:
        absa_trainer = Seq2SeqTrainerWrapper(
            model=model,
            tokenizer=tokenizer,
            dataset=tokenized_absa_dataset,
            **vars(args)
        )
        absa_trainer.train()

    # Step 7: Evaluate ABSA model
    if args.eval_absa:
        absa_predictor = Predictor(
            args.absa_model_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_beams=args.num_beams
        )
        absa_test_inputs = tokenized_absa_dataset["test"]
        absa_test_refs = tokenized_absa_dataset["test"]["target_text"]
        absa_preds = absa_predictor.predict(absa_test_inputs)
        absa_preds = [p[0].strip("[]").strip("'\"") if isinstance(p, list) else p.strip("[]").strip("'\"") for p in absa_preds]

        os.makedirs(args.save_dir, exist_ok=True)
        df = pd.DataFrame({
            "input": tokenized_absa_dataset["test"]['input_text'],
            "reference": absa_test_refs,
            "prediction": absa_preds
        })
        df.to_csv(os.path.join(args.save_dir, "absa_extraction_predictions.csv"), index=False)

        absa_eval_path = os.path.join(args.save_dir, "absa_eval_results.json")
        evaluate_absa_outputs(absa_test_refs, absa_preds, output_file=absa_eval_path)
        evaluate_absa_detailed(absa_test_refs, absa_preds, output_file="absa_detailed_metrics.json")

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

    current_file = Path(__file__).resolve()
    while current_file.name != "Aspect-Based-Sentiment-Analysis-for-Online-Movie-Reviews":
        current_file = current_file.parent
    ROOT_DIR = current_file

    

    def is_hf_model_name(model_path):
        return isinstance(model_path, str) and (
            model_path.startswith("t5") or
            model_path.startswith("facebook/") or
            model_path.startswith("google/") or
            model_path.count("/") == 1  # namespace/model
        )
    
    parser.add_argument("--prepare_data", action="store_true", help="Prepare and split dataset")
    parser.add_argument("--train_component", action="store_true", help="Train component extraction model")
    parser.add_argument("--eval_component", action="store_true", help="Evaluate component extraction model")
    parser.add_argument("--train_absa", action="store_true", help="Train ABSA model")
    parser.add_argument("--eval_absa", action="store_true", help="Evaluate ABSA model")
    parser.add_argument("--run_pipeline", action="store_true", help="Run end-to-end pipeline")

    parser.add_argument("--raw_data_path", type=str, default="data/raw/full_dataset.csv")
    parser.add_argument("--save_dir", type=str, default="data/processed")
    parser.add_argument("--component_model_path", type=str, default="models/t5_component_extraction")
    parser.add_argument("--absa_model_path", type=str, default="models/t5_absa")
    parser.add_argument("--pipeline_input", type=str, default="data/raw/full_reviews.csv")
    parser.add_argument("--pipeline_output", type=str, default="pipeline/output.csv")

    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Max checkpoints to keep")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging frequency in steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Checkpoint save strategy")
    parser.add_argument("--max_length", type=int, default=64, help="Max generation length for ABSA prediction")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search in ABSA prediction")

    args = parser.parse_args()



    if not is_hf_model_name(args.component_model_path):
        args.component_model_path = ROOT_DIR / args.component_model_path
    if not is_hf_model_name(args.absa_model_path):
        args.absa_model_path = ROOT_DIR / args.absa_model_path
    args.raw_data_path = ROOT_DIR / args.raw_data_path
    args.save_dir = ROOT_DIR / args.save_dir
    args.pipeline_input = ROOT_DIR / args.pipeline_input
    args.pipeline_output = ROOT_DIR / args.pipeline_output

    main(args)
