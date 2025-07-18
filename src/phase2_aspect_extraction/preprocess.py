# import torch

# def preprocess_absa(examples):
#     # Create input_text by prepending a prompt to each component sentence.
#     # This tells the model what task to perform.
#     # inputs = ["Extract aspect, sentiment, and opinion term: " + str(sentence)
#     inputs = ["Extract aspect, aspect term, sentiment, and opinion term from: " + str(sentence)
#               for sentence in examples["Component sentence"]]

#     # Create target_text by formatting the output structure using all relevant fields.
#     # If any field is missing (empty), replace it with "None".
#     targets = [
#         "aspect: " + (aspect if aspect else "None") +
#         ", aspect term: " + (aspect_term if aspect_term else "None") +
#         ", opinion term: " + (opinion if opinion else "None") +
#         ", sentiment: " + (sentiment if sentiment else "None")
#         for aspect, aspect_term, opinion, sentiment in zip(
#             examples["Aspect Category"],
#             examples["Aspect Terms"],
#             examples["Opinion Terms"],
#             examples["Polarity Sentiment"]
#         )
#     ]

#     return {"input_text": inputs, "target_text": targets}


# def tokenize_absa(batch, tokenizer):
#     # Tokenize the input_texts: convert to input_ids and attention masks
#     model_inputs = tokenizer(
#         batch["input_text"],
#         padding="max_length",
#         truncation=True,
#         max_length=128
#     )

#     # Tokenize the target_texts (labels)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(
#             batch["target_text"],
#             padding="max_length",
#             truncation=True,
#             max_length=128
#         )

#     # Replace padding token IDs in labels with -100 so they are ignored during loss computation
#     labels["input_ids"] = [
#         [(token if token != tokenizer.pad_token_id else -100) for token in label]
#         for label in labels["input_ids"]
#     ]

#     # If labels are scalar (1D tensor), unsqueeze to make them batch-like (for multi-GPU compatibility)
#     if isinstance(labels["input_ids"], torch.Tensor):
#         if labels["input_ids"].dim() == 1:
#             labels["input_ids"] = labels["input_ids"].unsqueeze(0)

#     # Add labels to the input dictionary
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

import torch

def preprocess_absa(examples, model_type="t5"):
    """
    Prepares input_text and target_text for Aspect-Based Sentiment Analysis (ABSA).
    Supports both T5 (with prompt) and BART (raw sentence).
    
    Args:
        examples (dict): Input batch with keys 'Component sentence', 'Aspect Category', etc.
        model_type (str): Either 't5' or 'bart'. T5 uses task prompt; BART uses raw text.
    
    Returns:
        dict: A dictionary with 'input_text' and 'target_text' fields.
    """
    if model_type.lower().startswith("t5"):
        inputs = [
            "Extract aspect, aspect term, sentiment, and opinion term from: " + str(sentence)
            for sentence in examples["Component sentence"]
        ]
    else:
        inputs = [str(sentence) for sentence in examples["Component sentence"]]

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


def tokenize_absa(batch, tokenizer, max_input_len=128, max_target_len=128):
    """
    Tokenizes input_text and target_text for ABSA task.
    Automatically adapts to T5 (with special tokenizer handling) and BART.
    
    Args:
        batch (dict): A batch with 'input_text' and 'target_text'
        tokenizer: HuggingFace tokenizer (T5Tokenizer or BartTokenizer)
        max_input_len (int): Max length for input tokens
        max_target_len (int): Max length for target tokens
    
    Returns:
        dict: Tokenized input dictionary including masked 'labels'
    """
    model_inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=max_input_len
    )

    # T5 uses special target tokenizer context
    if "t5" in tokenizer.name_or_path.lower():
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                padding="max_length",
                truncation=True,
                max_length=max_target_len
            )
    else:
        labels = tokenizer(
            batch["target_text"],
            padding="max_length",
            truncation=True,
            max_length=max_target_len
        )

    # Mask pad tokens in labels
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
