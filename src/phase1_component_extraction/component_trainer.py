# component_trainer.py
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import os

class ComponentTrainer:
    def __init__(self, dataset, model_name="t5-base", output_dir=".models/t5_component_extraction"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.dataset = dataset
        self.output_dir = output_dir

    def train(self, epochs=20, batch_size=4, lr=3e-4):
        os.environ["WANDB_DISABLED"] = "true"

        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            save_strategy="epoch",
            save_total_limit=1,
            logging_dir='./logs',
            logging_steps=50,
            learning_rate=lr,
            warmup_steps=500,
            weight_decay=0.01,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, self.model)
        )

        trainer.train()
