import os
import torch
import pandas as pd
from tqdm import tqdm

from src.phase1_component_extraction.component_predictor import ComponentPredictor
from src.phase2_aspect_extraction.absa_predictor import ABSAPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def run_absa_pipeline(
    input_csv: str,
    output_csv: str,
    component_model_path: str,
    absa_model_path: str
):
    """Run end-to-end component + ABSA prediction pipeline and save to CSV."""
    # Load reviews
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    if "review" not in df.columns:
        raise ValueError("Input CSV must contain a 'review' column.")
    full_reviews = df["review"].tolist()

    # Load models
    component_predictor = ComponentPredictor(component_model_path)
    absa_predictor = ABSAPredictor(absa_model_path)

    results = []
    for review in tqdm(full_reviews, desc="Running ABSA pipeline"):
        comp_sentences = component_predictor.predict_single([review])[0]
        
        for comp in comp_sentences:
            absa_output = absa_predictor.predict_single([comp])[0]
            parsed = parse_absa_output(absa_output)
            results.append({
                "full_review": review,
                "component": comp,
                "aspect": parsed["aspect"],
                "aspect_term": parsed["aspect_term"],
                "opinion_term": parsed["opinion_term"],
                "sentiment": parsed["sentiment"]
            })

    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved pipeline result to {output_csv}")
