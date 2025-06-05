import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

class ABSAPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)

    def predict(self, sentences, max_length=64, num_beams=4, batch_size=8):
        self.model.eval()
        inputs = ["Extract aspect, sentiment, and opinion term: " + s for s in sentences]
        encoded = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)

        predictions = []
        for i in tqdm(range(0, len(inputs), batch_size)):
            input_ids = encoded["input_ids"][i:i+batch_size]
            attention_mask = encoded["attention_mask"][i:i+batch_size]
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            predictions.extend(decoded)
        return predictions
