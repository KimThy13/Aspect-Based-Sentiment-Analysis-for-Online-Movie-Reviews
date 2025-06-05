import torch
import pandas as pd
from tqdm import tqdm

from phase1_component_extraction.component_predictor import ComponentPredictor
from phase2_aspect_extraction.absa_predictor import ABSAPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
component_model_path = "models/t5_component_extraction"
absa_model_path = "models/t5_absa"

component_predictor = ComponentPredictor(component_model_path)
absa_predictor = ABSAPredictor(absa_model_path)

def absa_pipeline(full_review):
    # Step 1: Extract component sentences
    component_sentences = component_predictor.predict([full_review])[0]  # trả về list[list[str]]

    results = []
    # Step 2: ABSA for each sentence
    for comp in component_sentences:
        absa_output = absa_predictor.predict([comp])[0]
        results.append({
            "full_review": full_review,
            "component": comp,
            "absa_output": absa_output
        })

    return pd.DataFrame(results)

def process_reviews(full_reviews):
    all_results = []
    for review in tqdm(full_reviews, desc="Running End-to-End ABSA"):
        result_df = absa_pipeline(review)
        all_results.append(result_df)
    return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":
    df = pd.read_csv("data/raw/full_reviews.csv")
    full_reviews = df["review"].tolist()

    final_result = process_reviews(full_reviews)
    final_result.to_csv("pipeline/output.csv", index=False)
    print("✅ Saved pipeline/output.csv")
