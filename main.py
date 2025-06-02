from preprocessing.component_preprocessing import preprocess_component_batch, tokenize_component, tokenizer

# Bước 1: preprocess từng batch (gộp câu theo review)
processed_component = dataset.map(
    preprocess_component_batch,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Bước 2: tokenize input và target
tokenized_component = processed_component.map(
    lambda batch: tokenize_component(batch, tokenizer),
    batched=True
)
