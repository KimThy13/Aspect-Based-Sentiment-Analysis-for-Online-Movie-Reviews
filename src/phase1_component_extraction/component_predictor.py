# component_predictor.py
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ComponentPredictor:
    def __init__(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, inputs, batch_size=8, max_length=512, num_beams=4):
        tokenized = self.tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)

        predictions = []
        for i in tqdm(range(0, len(input_ids), batch_size)):
            batch_ids = input_ids[i:i + batch_size]
            with torch.no_grad():
                outputs = self.model.generate(
                    batch_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            decoded = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
            predictions.extend(decoded)

        return predictions
