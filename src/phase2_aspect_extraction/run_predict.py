from utils.data_loader import load_absa_dataset
from absa_predictor import ABSAPredictor
from utils.evaluator import evaluate_absa_outputs

def run_absa_prediction(model_path, data_path="data/preprocessed/absa"):
    # Load dataset đã preprocess
    dataset = load_absa_dataset(data_path)

    # Load mô hình
    predictor = ABSAPredictor(model_path)

    # Chuẩn bị input (từ câu component)
    test_sentences = dataset["test"]["Component sentence"]

    # Sinh dự đoán
    predictions = predictor.predict(test_sentences)

    # Nếu có target_text thì đánh giá luôn
    if "target_text" in dataset["test"].column_names:
        target_texts = dataset["test"]["target_text"]
        evaluate_absa_outputs(target_texts, predictions)

    with open("absa_predictions.txt", "w", encoding="utf-8") as f:
        for line in predictions:
            f.write(line + "\n")

    # Trả về dự đoán nếu cần lưu/log
    return predictions


if __name__ == "__main__":
    run_absa_prediction(model_path="models/t5_absa")
