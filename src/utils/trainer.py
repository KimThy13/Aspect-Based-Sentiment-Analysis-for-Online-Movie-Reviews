# src/utils/trainer.py
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import os

class T5Trainer:
    def __init__(self, dataset, **kwargs):
        self.model_name = kwargs.get("model_name", "t5-base")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.dataset = dataset
        
        self.output_dir = kwargs.get("output_dir", ".models/t5_trainer")
        self.epochs = kwargs.get("epochs", 20)
        self.batch_size = kwargs.get("batch_size", 8)
        self.lr = kwargs.get("lr", 3e-4)

        # Các tham số khác truyền thêm qua kwargs
        self.save_total_limit = kwargs.get('save_total_limit', 1)
        self.logging_steps = kwargs.get('logging_steps', 50)
        self.warmup_steps = kwargs.get('warmup_steps', 500)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.save_strategy = kwargs.get('save_strategy', 'epoch')

    def train(self):
        os.environ["WANDB_DISABLED"] = "true"

        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            logging_dir='./logs',
            logging_steps=self.logging_steps,
            learning_rate=self.lr,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
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
