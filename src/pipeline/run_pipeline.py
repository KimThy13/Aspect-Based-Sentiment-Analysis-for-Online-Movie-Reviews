import torch
import pandas as pd
from tqdm import tqdm

from phase1_component_extraction.component_predictor import ComponentPredictor
from phase2_aspect_extraction.absa_predictor import ABSAPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained models for component extraction and ABSA
component_model_path = "models/t5_component_extraction"
absa_model_path = "models/t5_absa"

component_predictor = ComponentPredictor(component_model_path)
absa_predictor = ABSAPredictor(absa_model_path)

def absa_pipeline(full_review):
    # Step 1: Extract component sentences from the full review text
    component_sentences = component_predictor.predict([full_review])[0]  # returns list of component sentences

    results = []
    # Step 2: For each component sentence, run ABSA prediction
    for comp in component_sentences:
        absa_output = absa_predictor.predict([comp])[0]  # predict aspect, sentiment, opinion
        results.append({
            "full_review": full_review,   # original review text
            "component": comp,            # extracted component sentence
            "absa_output": absa_output    # ABSA model prediction on this component
        })

    # Return a DataFrame of results for this full review
    return pd.DataFrame(results)

def process_reviews(full_reviews):
    all_results = []
    # Iterate over all full reviews with progress bar
    for review in tqdm(full_reviews, desc="Running End-to-End ABSA"):
        # Run the full pipeline for each review and collect results
        result_df = absa_pipeline(review)
        all_results.append(result_df)

    # Concatenate all results into a single DataFrame
    return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":
    # Load raw reviews from CSV
    df = pd.read_csv("data/raw/full_reviews.csv")
    full_reviews = df["review"].tolist()

    # Run the pipeline on all reviews
    final_result = process_reviews(full_reviews)

    # Save the final results to CSV
    final_result.to_csv("pipeline/output.csv", index=False)
    print("Saved pipeline/output.csv")
