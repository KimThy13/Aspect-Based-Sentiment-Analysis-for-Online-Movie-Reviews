from collections import defaultdict
import torch

def preprocess_component(batch, model_type):
    """
    Prepare input-target pairs for component sentence extraction task.
    Supports both T5 and BART by adjusting prompt usage.
    
    Args:
        batch (dict): A batch of raw examples with 'Review ID', 'Full review', 'Component sentence'
        model_type (str): Either 't5' or 'bart'. T5 uses a task prompt; BART uses raw review.
    
    Returns:
        dict: A dictionary with 'input_text' and 'target_text'
    """
    grouped = defaultdict(lambda: {"full_review": "", "component_sentences": []})

    for i in range(len(batch["Review ID"])):
        review_id = batch["Review ID"][i].split("-")[0]
        grouped[review_id]["full_review"] = str(batch["Full review"][i])
        grouped[review_id]["component_sentences"].append(str(batch["Component sentence"][i]))

    inputs, targets = [], []

    for group in grouped.values():
        full_review = group["full_review"]
        target_text = " ; ".join(group["component_sentences"])

        if str(model_type).lower().startswith("t5"):
            input_text = f"Extract component sentences: {full_review}" #prompt for t5
        else:
            input_text = full_review  # No prompt for BART

        inputs.append(input_text)
        targets.append(target_text)

    return {"input_text": inputs, "target_text": targets}


def tokenize_component(batch, tokenizer, max_input_len=512, max_target_len=512):
    """
    Tokenize input_text and target_text for both T5 and BART.
    Automatically handles masking and tokenizer-specific behavior.
    
    Args:
        batch (dict): A batch with 'input_text' and 'target_text'
        tokenizer: HuggingFace tokenizer (T5Tokenizer or BartTokenizer)
        max_input_len (int): Max token length for input text
        max_target_len (int): Max token length for target text
    
    Returns:
        dict: Tokenized input with 'input_ids', 'attention_mask', and masked 'labels'
    """
    model_inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=max_input_len,
    )

    # Use target tokenization depending on tokenizer type
    if "t5" in tokenizer.name_or_path.lower():
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                padding="max_length",
                truncation=True,
                max_length=max_target_len,
            )
    else:
        labels = tokenizer(
            batch["target_text"],
            padding="max_length",
            truncation=True,
            max_length=max_target_len,
        )

    # Replace pad token IDs with -100 to ignore them during loss computation
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
