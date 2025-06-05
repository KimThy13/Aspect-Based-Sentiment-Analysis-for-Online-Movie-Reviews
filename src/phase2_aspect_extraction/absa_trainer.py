from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, T5ForConditionalGeneration
import os

class ABSATrainer:
    def __init__(self, tokenizer, dataset, model_name="t5-base"):
        # Store the tokenizer and dataset
        self.tokenizer = tokenizer
        self.dataset = dataset

        # Load the T5 model from the given pretrained name (e.g., "t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def train(self, output_dir=".models/t5_absa", epochs=10, batch_size=4):
        # Disable Weights & Biases logging
        os.environ["WANDB_DISABLED"] = "true"

        # Define training arguments for the Seq2SeqTrainer
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,                       # Directory to save checkpoints and final model
            per_device_train_batch_size=batch_size,      # Batch size for training
            per_device_eval_batch_size=batch_size,       # Batch size for evaluation
            num_train_epochs=epochs,                     # Number of training epochs
            logging_dir='./logs',                        # Directory to save training logs
            save_total_limit=1,                          # Only keep the latest saved model
            save_strategy="epoch",                       # Save model at the end of each epoch
            logging_steps=50,                            # Log training info every 50 steps
            learning_rate=3e-4,                          # Initial learning rate
            warmup_steps=500,                            # Warmup steps before applying learning rate schedule
            weight_decay=0.01                            # Weight decay to prevent overfitting
        )

        # Create a Trainer to manage the training loop
        trainer = Seq2SeqTrainer(
            model=self.model,                             # The T5 model
            args=training_args,                           # Training configurations
            train_dataset=self.dataset["train"],          # Training data
            eval_dataset=self.dataset["validation"],      # Validation data
            tokenizer=self.tokenizer,                     # Tokenizer used for preprocessing
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, self.model)  # Handles batching/padding
        )

        # Start the training process
        trainer.train()
