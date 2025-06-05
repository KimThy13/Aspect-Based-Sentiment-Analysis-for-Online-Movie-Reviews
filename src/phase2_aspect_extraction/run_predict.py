from utils.data_loader import load_absa_dataset
from absa_predictor import ABSAPredictor
from utils.evaluator import evaluate_absa_outputs

def run_absa_prediction(model_path, data_path="data/preprocessed/absa"):
    # Load the preprocessed/test ABSA dataset from disk
    dataset = load_absa_dataset(data_path)

    # Initialize the ABSA predictor with the trained model
    predictor = ABSAPredictor(model_path)

    # Prepare the input sentences (usually component-level sentences)
    test_sentences = dataset["test"]["Component sentence"]

    # Generate predictions using the model
    predictions = predictor.predict(test_sentences)

    # If the test set contains ground-truth targets, evaluate predictions
    if "target_text" in dataset["test"].column_names:
        target_texts = dataset["test"]["target_text"]
        evaluate_absa_outputs(target_texts, predictions)

    # Save the predictions to a text file for later analysis
    with open("absa_predictions.txt", "w", encoding="utf-8") as f:
        for line in predictions:
            f.write(line + "\n")

    # Return predictions if needed (e.g., for logging or further use)
    return predictions

# Run prediction if this file is executed directly
if __name__ == "__main__":
    run_absa_prediction(model_path="models/t5_absa")
