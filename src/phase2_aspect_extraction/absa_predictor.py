import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import default_data_collator

class ABSAPredictor:
    def __init__(self, model_path, max_length=64, num_beams=4, batch_size=8):
        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model from the specified path
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)

        # Store generation parameters as instance variables
        self.max_length = max_length
        self.num_beams = num_beams
        self.batch_size = batch_size

    def predict(self, tokenized_dataset):
        predictions = []

        # Sử dụng DataLoader để chia batch và tự xử lý padding
        dataloader = DataLoader(
            tokenized_dataset, 
            batch_size=self.batch_size, 
            collate_fn=default_data_collator
        )

        self.model.eval()

        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                    num_return_sequences=1,
                    do_sample=False
                )

            # Decode predictions
            decoded_preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(decoded_preds)

        return predictions
    
    def predict_single(self, input_text):
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
        
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded
        