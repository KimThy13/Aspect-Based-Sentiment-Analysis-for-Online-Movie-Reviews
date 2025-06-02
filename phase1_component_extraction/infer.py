import torch
from tqdm import tqdm

def generate_predictions_task1(model, tokenizer, raw_dataset, batch_size=8, max_length=512, num_beams=4):
    """
    Hàm tạo dự đoán từ tập dữ liệu test sử dụng mô hình và tokenizer đã huấn luyện.
    
    Tham số:
    - model: Mô hình đã fine-tune
    - tokenizer: Tokenizer tương ứng
    - raw_dataset: dataset chưa token hóa, có field "input_text"
    
    Trả về:
    - predictions: Danh sách các câu thành phần được dự đoán (list[str])
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = raw_dataset["test"]["input_text"]
    predictions = []

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_texts = inputs[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predictions.extend(decoded)

    return predictions
