import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

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

    def predict(self, sentences):
        # Set the model to evaluation mode (no training)
        self.model.eval()

        # Prepare inputs by adding a task prompt to each sentence
        inputs = ["Extract aspect, sentiment, aspect term and opinion term: " + s for s in sentences]

        # Tokenize the inputs and move them to the correct device
        encoded = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        predictions = []

        # Process inputs in batches
        for i in tqdm(range(0, len(inputs), self.batch_size)):
            input_ids = encoded["input_ids"][i:i+self.batch_size]
            attention_mask = encoded["attention_mask"][i:i+self.batch_size]

            # Inference without computing gradients
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )

            # Decode generated token IDs into strings and collect predictions
            decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            predictions.extend(decoded)

        return predictions
