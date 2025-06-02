import os
import sys

# Thêm path tới thư mục 'data/' để import load_and_split_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from preprocessing import preprocess_component_batch, tokenize_component
from load_dataset import load_and_split_dataset  # từ file data/load_dataset.py

# Tắt wandb
os.environ["WANDB_DISABLED"] = "true"

# ---------------------- Load & Split Dataset ----------------------
dataset = load_and_split_dataset(
    csv_path="data/raw/component.csv",  # đường dẫn tới file CSV
    id_column="Review ID",              # hoặc tên cột ID đúng trong file
    test_size=0.2,                      # 20% để chia val và test
    val_size=0.1,                       # tách thêm validation
    random_state=42
)

# ---------------------- Tiền xử lý ----------------------
# Tạo input_text và target_text
processed_dataset = dataset.map(preprocess_component_batch, batched=True, remove_columns=dataset["train"].column_names)

# Tokenize
tokenized_dataset = processed_dataset.map(tokenize_component, batched=True)

# ---------------------- Load Model ----------------------
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# ---------------------- Huấn luyện ----------------------
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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

# ---------------------- Train ----------------------
if __name__ == "__main__":
    trainer.train()
