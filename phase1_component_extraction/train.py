import os
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
)
from datasets import load_dataset
from phase1_component_extraction.preprocessing import preprocess_component_batch, tokenize_component

os.environ["WANDB_DISABLED"] = "true"

# Load raw dataset
dataset = load_dataset("csv", data_files={"train": "data/raw/component.csv"}, split="train")

# Train/Validation split
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Preprocess
processed = dataset.map(preprocess_component_batch, batched=True, remove_columns=dataset["train"].column_names)
tokenized = processed.map(tokenize_component, batched=True)

# Load model + tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/t5_component_phase1",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    learning_rate=3e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

# ---------------------- Train ----------------------
if __name__ == "__main__":
    trainer.train()
