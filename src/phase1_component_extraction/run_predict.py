from utils.data_loader import load_component_dataset
from component_predictor import ComponentPredictor
from utils.evaluator import evaluate_component_outputs

def run_component_prediction(model_path, data_path="data/preprocessed/component"):
    # Load dataset đã preprocess/tokenize
    dataset = load_component_dataset(data_path)

    # Load mô hình
    predictor = ComponentPredictor(model_path)

    # Chuẩn bị input text
    test_sentences = dataset["test"]["input_text"]

    # Sinh dự đoán
    predictions = predictor.predict(test_sentences)

    # Nếu có nhãn thì đánh giá luôn
    if "target_text" in dataset["test"].column_names:
        target_texts = dataset["test"]["target_text"]
        evaluate_component_outputs(target_texts, predictions)

    # Lưu lại kết quả
    with open("component_predictions.txt", "w", encoding="utf-8") as f:
        for line in predictions:
            f.write(line + "\n")

    return predictions

if __name__ == "__main__":
    run_component_prediction(model_path="models/t5_component_extraction")
