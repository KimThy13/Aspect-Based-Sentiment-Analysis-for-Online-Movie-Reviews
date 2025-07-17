import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import default_data_collator


class Predictor:
    def __init__(self, model_path, max_length=64, num_beams=4, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detect and load model/tokenizer
        if "t5" in str(model_path).lower():
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        elif "bart" in (model_path).lower():
            # self.tokenizer = BartTokenizer.from_pretrained(model_path)
            # self.model = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type in {model_path}")

        self.max_length = max_length
        self.num_beams = num_beams
        self.batch_size = batch_size

    def predict(self, tokenized_dataset):
        predictions = []

        if "input_text" not in tokenized_dataset.column_names:
            raise ValueError("`input_text` must be present in the dataset.")

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
