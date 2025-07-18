import os
import torch
import pandas as pd
from tqdm import tqdm
import chardet
from datasets import Dataset

from src.utils.predictor import Predictor
from transformers import default_data_collator


def parse_absa_output(output_str):
    """Convert ABSA output string to a structured dictionary."""
    try:
        parts = {
            "aspect": output_str.split("aspect:")[1].split(",")[0].strip(),
            "aspect_term": output_str.split("aspect term:")[1].split(",")[0].strip(),
            "opinion_term": output_str.split("opinion term:")[1].split(",")[0].strip(),
            "sentiment": output_str.split("sentiment:")[1].strip()
        }
        return parts
    except Exception:
        return {"aspect": None, "aspect_term": None, "opinion_term": None, "sentiment": None}


def detect_model_type(model_path_or_name):
    """Detects whether the model is T5 or BART based on its name or path."""
    name = str(model_path_or_name).lower()
    if "t5" in name:
        return "t5"
    elif "bart" in name:
        return "bart"
    else:
        raise ValueError(f"Cannot detect model type from path: {model_path_or_name}")


def prepare_component_inputs(reviews, tokenizer, model_type="t5", max_len=512):
    """
    Tokenizes input reviews for component extraction phase.
    Uses task prompt only if model is T5.
    """
    inputs = []
    for r in reviews:
        cleaned = str(r).replace("\n", " ").strip()
        if model_type == "t5":
            inputs.append(f"Extract component sentences: {cleaned}")
        else:
            inputs.append(cleaned)

    enc = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    dataset_dict = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "input_text": inputs
    }

    return Dataset.from_dict(dataset_dict)


def build_absa_prompt(text: str, model_type: str) -> str:
    """
    Builds the input prompt for ABSA prediction.
    For T5: uses a task-specific instruction.
    For BART: uses the raw sentence.
    """
    if model_type == "t5":
        return f"Extract aspect, aspect term, sentiment, and opinion term from: {text.strip()}"
    else:
        return text.strip()


def run_absa_pipeline(
    input_csv: str,
    output_csv: str,
    component_model_path: str,
    absa_model_path: str,
    max_len: int = 512,
):
    # Detect encoding of input CSV file
    with open(input_csv, "rb") as f:
        raw = f.read(100_000)
        encoding = chardet.detect(raw)["encoding"] or "utf-8-sig"

    df = pd.read_csv(input_csv, encoding=encoding)
    if "review" not in df.columns:
        raise ValueError("Input CSV must contain a 'review' column.")

    reviews = df["review"].fillna("").tolist()

    # Load models
    component_predictor = Predictor(component_model_path)
    absa_predictor = Predictor(absa_model_path)

    # Detect model types
    component_model_type = detect_model_type(component_model_path)
    absa_model_type = detect_model_type(absa_model_path)

    # Prepare input for component prediction
    tokenized_dataset = prepare_component_inputs(
        reviews, component_predictor.tokenizer, model_type=component_model_type, max_len=max_len
    )

    # Run component sentence prediction
    comp_outputs = component_predictor.predict(tokenized_dataset)  # List[List[str]]

    results = []
    for full_review, comp_list in tqdm(
        zip(reviews, comp_outputs), total=len(reviews), desc="ABSA pipeline"
    ):
        for comp in comp_list:
            absa_prompt = build_absa_prompt(comp, absa_model_type)
            absa_pred = absa_predictor.predict_single(absa_prompt)
            parsed = parse_absa_output(absa_pred)
            results.append({
                "full_review": full_review,
                "component": comp,
                "aspect": parsed["aspect"],
                "aspect_term": parsed["aspect_term"],
                "opinion_term": parsed["opinion_term"],
                "sentiment": parsed["sentiment"]
            })

    # Save results to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
