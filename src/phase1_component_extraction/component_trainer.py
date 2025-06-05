from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import os

class ComponentTrainer:
    def __init__(self, dataset, model_name="t5-base", output_dir=".models/t5_component_extraction"):
        # Load the tokenizer and model using the given model name (e.g., "t5-base")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Store the dataset (should be a dictionary with "train" and "validation" splits)
        self.dataset = dataset

        # Set the output directory for saving the trained model
        self.output_dir = output_dir

    def train(self, epochs=20, batch_size=4, lr=3e-4):
        # Disable Weights & Biases logging
        os.environ["WANDB_DISABLED"] = "true"

        # Define training arguments
        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,                     # Directory to save checkpoints
            per_device_train_batch_size=batch_size,         # Training batch size per device
            per_device_eval_batch_size=batch_size,          # Evaluation batch size per device
            num_train_epochs=epochs,                        # Number of training epochs
            save_strategy="epoch",                          # Save a checkpoint after every epoch
            save_total_limit=1,                             # Keep only the most recent checkpoint
            logging_dir='./logs',                           # Directory for logging
            logging_steps=50,                               # Log every 50 steps
            learning_rate=lr,                               # Learning rate
            warmup_steps=500,                               # Warmup steps for the learning rate scheduler
            weight_decay=0.01,                              # Weight decay to reduce overfitting
        )

        # Initialize the Trainer with model, arguments, datasets, tokenizer, and data collator
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, self.model)  # Handles padding and batching
        )

        # Start the training process
        trainer.train()
