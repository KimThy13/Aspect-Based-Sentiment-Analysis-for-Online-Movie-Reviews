from collections import defaultdict
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")


def preprocess_absa(examples):
    # Tạo input_text từ câu
    ### Với mỗi câu, thêm prefix "Extract aspect, sentiment, and opinion term: " để hướng mô hình làm đúng nhiệm vụ.
    inputs = ["Extract aspect, sentiment, and opinion term: " + str(sentence) for sentence in examples["Component sentence"]]

    # Tạo target_text từ các cột aspect, aspect term, opinion term, sentiment
    targets = [
        "aspect: " + (aspect if aspect else "None") +
        ", aspect term: " + (aspect_term if aspect_term else "None") +
        ", opinion term: " + (opinion if opinion else "None") +
        ", sentiment: " + (sentiment if sentiment else "None")
        for aspect, aspect_term, opinion, sentiment in zip(
            examples["Aspect Category"],
            examples["Aspect Terms"],
            examples["Opinion Terms"],
            examples["Polarity Sentiment"]
        )
    ]
    return {"input_text": inputs, "target_text": targets}

def tokenize_absa(batch):
    # Tokenize input
    model_inputs = tokenizer(batch["input_text"],padding="max_length",truncation=True,max_length=128)

    # Tokenize target
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target_text"],padding="max_length",truncation=True,max_length=128)

    # Replace padding token id in labels with -100 to ignore in loss
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    # Unsqueeze the labels if they are scalar (for multi-GPU)
    if isinstance(labels["input_ids"], torch.Tensor):
        if labels["input_ids"].dim() == 1:
            labels["input_ids"] = labels["input_ids"].unsqueeze(0)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs