# Import necessary libraries
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ComponentPredictor:
    def __init__(self, model_path):
        # Load the tokenizer and model from the given path
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Use GPU if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move the model to the selected device and set it to evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def predict(self, inputs, batch_size=8, max_length=512, num_beams=4):
        # Tokenize the input texts with padding and truncation
        tokenized = self.tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        
        # Move tokenized input IDs to the selected device
        input_ids = tokenized["input_ids"].to(self.device)

        predictions = []

        # Process the inputs in batches
        for i in tqdm(range(0, len(input_ids), batch_size)):
            batch_ids = input_ids[i:i + batch_size]
            
            # Disable gradient calculation for inference
            with torch.no_grad():
                # Generate output sequences using beam search
                outputs = self.model.generate(
                    batch_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )

            # Decode the generated token IDs into text and add to predictions
            decoded = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
            predictions.extend(decoded)

        # Return the list of predicted output strings
        return predictions
