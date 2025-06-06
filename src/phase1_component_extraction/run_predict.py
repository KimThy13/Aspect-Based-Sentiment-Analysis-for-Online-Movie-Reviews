from utils.data_loader import load_dataset_from_folder
from component_predictor import ComponentPredictor
from utils.evaluator import evaluate_component_outputs

def run_component_prediction(model_path, data_path="data/preprocessed/component"):
    # Load the preprocessed/tokenized dataset from the given path
    dataset = load_dataset_from_folder(data_path)

    # Load the trained model using the ComponentPredictor class
    predictor = ComponentPredictor(model_path)

    # Prepare input texts from the test set
    test_sentences = dataset["test"]["input_text"]

    # Generate predictions for the test inputs
    predictions = predictor.predict(test_sentences)

    # If target labels exist in the dataset, evaluate the predictions
    if "target_text" in dataset["test"].column_names:
        target_texts = dataset["test"]["target_text"]
        evaluate_component_outputs(target_texts, predictions)

    # Save the predictions to a text file
    with open("component_predictions.txt", "w", encoding="utf-8") as f:
        for line in predictions:
            f.write(line + "\n")

    return predictions

# Run prediction when this script is executed directly
if __name__ == "__main__":
    run_component_prediction(model_path="models/t5_component_extraction")
