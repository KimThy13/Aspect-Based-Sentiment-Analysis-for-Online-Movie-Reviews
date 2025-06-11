from collections import defaultdict
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def preprocess_component(batch):
    """
    Groups component sentences by review ID and prepares input-target pairs
    for training a model to extract component sentences from full reviews.
    """
    # Create a dictionary to group data by Review ID
    grouped = defaultdict(lambda: {"full_review": "", "component_sentences": []})

    # Group the full review and associated component sentences by review ID
    for i in range(len(batch["Review ID"])):
        review_id = batch["Review ID"][i].split("-")[0]  # Take base review ID
        grouped[review_id]["full_review"] = str(batch["Full review"][i])
        grouped[review_id]["component_sentences"].append(str(batch["Component sentence"][i]))

    inputs, targets = [], []

    # Create input-target pairs
    for group in grouped.values():
        # Input: full review with a task prompt
        input_text = f"Extract component sentences: {group['full_review']}"
        # Target: all component sentences separated by semicolons
        target_text = " ; ".join(group["component_sentences"])
        inputs.append(input_text)
        targets.append(target_text)

    return {"input_text": inputs, "target_text": targets}

def tokenize_component(batch, tokenizer):
    """
    Tokenizes the input and target texts using the T5 tokenizer.
    Labels (targets) are masked where padding tokens are present (set to -100).
    """
    # Tokenize the input text
    model_inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=512)

    # Tokenize the target text (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=512)

    # Replace pad token IDs with -100 so they are ignored in the loss computation
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    # Handle case where label input_ids might be scalar (e.g., for multi-GPU training)
    if isinstance(labels["input_ids"], torch.Tensor):
        if labels["input_ids"].dim() == 1:
            labels["input_ids"] = labels["input_ids"].unsqueeze(0)

    # Add the labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
