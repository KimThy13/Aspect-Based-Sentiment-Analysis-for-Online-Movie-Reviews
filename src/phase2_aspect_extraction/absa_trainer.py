from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, T5ForConditionalGeneration
import os

class ABSATrainer:
    def __init__(self, tokenizer, dataset, model_name="t5-base"):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def train(self, output_dir=".models/t5_absa", epochs=10, batch_size=4):
        os.environ["WANDB_DISABLED"] = "true"
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir='./logs',
            save_total_limit=1,
            save_strategy="epoch",
            logging_steps=50,
            learning_rate=3e-4,
            warmup_steps=500,
            weight_decay=0.01,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, self.model),
        )
        trainer.train()
