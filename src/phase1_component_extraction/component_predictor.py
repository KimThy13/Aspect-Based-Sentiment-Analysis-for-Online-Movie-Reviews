# Import necessary libraries
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import default_data_collator

class ComponentPredictor:
    def __init__(self, model_path, max_length=512, num_beams=4, batch_size=8):
        # Load the tokenizer and model from the given path
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, use_safetensors=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, use_safetensors=True)
        
        # Use GPU if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move the model to the selected device and set it to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Store generation parameters as instance variables
        self.max_length = max_length
        self.num_beams = num_beams
        self.batch_size = batch_size

    def predict(self, tokenized_dataset):
        predictions = []

        if "input_text" not in tokenized_dataset.column_names:
            raise ValueError("`input_text` must be present in the dataset for filtering.")

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
                    num_return_sequences=1
                )

            decoded_batch = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for pred in decoded_batch:
                comps = [x.strip() for x in pred.split(";") if x.strip()]
                predictions.append(comps)

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
