from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import os

class Seq2SeqTrainerWrapper:
    def __init__(self, model, tokenizer, dataset, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.output_dir = kwargs.get("output_dir", "./models/trainer")
        self.epochs = kwargs.get("epochs", 20)
        self.batch_size = kwargs.get("batch_size", 8)
        self.lr = kwargs.get("lr", 3e-4)

        self.save_total_limit = kwargs.get('save_total_limit', 1)
        self.logging_steps = kwargs.get('logging_steps', 50)
        self.warmup_steps = kwargs.get('warmup_steps', 500)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.save_strategy = kwargs.get('save_strategy', 'epoch')

    def train(self):
        os.environ["WANDB_DISABLED"] = "true"

        training_args = Seq2SeqTrainingArguments(
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
            # evaluation_strategy="epoch",  # optional: evaluate every epoch
            # predict_with_generate=True     # ensures decoding during eval
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()
