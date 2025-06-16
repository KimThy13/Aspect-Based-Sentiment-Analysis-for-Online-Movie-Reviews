from collections import defaultdict
import torch

def preprocess_absa(examples):
    # Create input_text by prepending a prompt to each component sentence.
    # This tells the model what task to perform.
    # inputs = ["Extract aspect, sentiment, and opinion term: " + str(sentence)
    inputs = ["Extract aspect, aspect term, sentiment, and opinion term from: " + str(sentence)
              for sentence in examples["Component sentence"]]

    # Create target_text by formatting the output structure using all relevant fields.
    # If any field is missing (empty), replace it with "None".
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


def tokenize_absa(batch, tokenizer):
    # Tokenize the input_texts: convert to input_ids and attention masks
    model_inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # Tokenize the target_texts (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # Replace padding token IDs in labels with -100 so they are ignored during loss computation
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    # If labels are scalar (1D tensor), unsqueeze to make them batch-like (for multi-GPU compatibility)
    if isinstance(labels["input_ids"], torch.Tensor):
        if labels["input_ids"].dim() == 1:
            labels["input_ids"] = labels["input_ids"].unsqueeze(0)

    # Add labels to the input dictionary
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
