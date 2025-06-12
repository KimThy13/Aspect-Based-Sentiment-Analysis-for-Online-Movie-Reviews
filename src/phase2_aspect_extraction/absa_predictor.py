import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

class ABSAPredictor:
    def __init__(self, model_path, max_length=64, num_beams=4, batch_size=8):
        # Load the tokenizer and model from the given path
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, use_safetensors=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, use_safetensors=True)

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

        # Duyệt theo từng batch
        for i in tqdm(range(0, len(tokenized_dataset), self.batch_size)):
            batch = tokenized_dataset[i:i+self.batch_size]

            # Tạo tensor input_ids và attention_mask từ batch
            input_ids = [torch.tensor(example["input_ids"]) for example in batch]
            attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]

            # Padding cho các sequence trong batch để bằng chiều nhau
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            # Disable gradient calculation for inference
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )

            # Decode the generated token IDs into text and add to predictions
            for output in output_ids:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                predictions.append(decoded)

        return predictions
    # def predict(self, sentences):
    #     # Set the model to evaluation mode (no training)
    #     self.model.eval()

    #     # Prepare inputs by adding a task prompt to each sentence
    #     inputs = ["Extract aspect, sentiment, aspect term and opinion term: " + s for s in sentences]

    #     # Tokenize the inputs and move them to the correct device
    #     encoded = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
    #     predictions = []

    #     # Process inputs in batches
    #     for i in tqdm(range(0, len(inputs), self.batch_size)):
    #         input_ids = encoded["input_ids"][i:i+self.batch_size]
    #         attention_mask = encoded["attention_mask"][i:i+self.batch_size]

    #         # Inference without computing gradients
    #         with torch.no_grad():
    #             output = self.model.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 max_length=self.max_length,
    #                 num_beams=self.num_beams,
    #                 early_stopping=True
    #             )

    #         # Decode generated token IDs into strings and collect predictions
    #         decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
    #         predictions.extend(decoded)

    #     return predictions
