import os
import torch
import pandas as pd
from tqdm import tqdm
import chardet

from src.utils.predictor import Predictor
from transformers import default_data_collator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datasets import Dataset

def parse_absa_output(output_str):
    """Convert ABSA output string to structured dict."""
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

# build component prompt and tokenize
def prepare_component_inputs(reviews, tokenizer, max_len=512):
    prompts = []
    for r in reviews:
        cleaned = str(r).replace("\n", " ").strip()
        prompts.append(f"Extract component sentences: {cleaned}") # prompt

    enc = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    dataset_dict = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "input_text": prompts,  # optional – nếu predict() cần check
    }

    return Dataset.from_dict(dataset_dict)

# build prompt for absa
def build_absa_prompt(text: str) -> str:
    return (
        f"Extract aspect, aspect term, sentiment, and opinion term from: {text.strip()}"
    )


# main pipeline including 2 phases
def run_absa_pipeline(
    input_csv: str,
    output_csv: str,
    component_model_path: str,
    absa_model_path: str,
    max_len: int = 512,
):
    # Read file csv and detect encoding
    with open(input_csv, "rb") as f:
        raw = f.read(100_000)
        encoding = chardet.detect(raw)["encoding"] or "utf-8-sig"

    df = pd.read_csv(input_csv, encoding=encoding)
    if "review" not in df.columns:
        raise ValueError("Input CSV must contain a 'review' column.")

    reviews = df["review"].fillna("").tolist()

    # Load model (read from model path)
    component_predictor = Predictor(component_model_path)
    absa_predictor = Predictor(absa_model_path)

    # Prepare dataset for component model
    tokenized_dataset = prepare_component_inputs(
        reviews, component_predictor.tokenizer, max_len=max_len
    )

    # Predict component sentences
    comp_outputs = component_predictor.predict(tokenized_dataset)  # list[list[str]]

    results = []
    for full_review, comp_list in tqdm(
        zip(reviews, comp_outputs), total=len(reviews), desc="ABSA pipeline"
    ):
        # comp_list is the list of component sentences in a full review
        for comp in comp_list:
            # Prompt ABSA
            absa_prompt = build_absa_prompt(comp)
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

    # saving the result
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
