from collections import defaultdict
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def preprocess_component(batch):
    grouped = defaultdict(lambda: {"full_review": "", "component_sentences": []})

    for i in range(len(batch["Review ID"])):
        review_id = batch["Review ID"][i].split("-")[0]
        grouped[review_id]["full_review"] = str(batch["Full review"][i])
        grouped[review_id]["component_sentences"].append(str(batch["Component sentence"][i]))

    inputs, targets = [], []
    for group in grouped.values():
        input_text = f"Extract component sentences: {group['full_review']}"
        target_text = " ; ".join(group["component_sentences"])
        inputs.append(input_text)
        targets.append(target_text)

    return {"input_text": inputs, "target_text": targets}


def tokenize_component(batch, tokenizer):
    model_inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=512)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=512)

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
