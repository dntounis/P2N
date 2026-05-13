import argparse
from transformers import VisionEncoderDecoderModel, DonutProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset import P2NDataset, collate_fn

def train(dataset_path, output_dir, epochs, batch_size):
    # Load model and processor
    model_id = "naver-clova-ix/donut-base"
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Add special token for our task
    new_tokens = ["<s_p2n>"]
    processor.tokenizer.add_tokens(new_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Model configuration for generation
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_p2n>"])[0]
    # Set max length to match our dataset
    model.config.max_length = 512

    # Load dataset
    train_dataset = P2NDataset(dataset_path, processor, split="train")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=1,
        save_total_limit=1,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train P2N model.")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="./p2n-model", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    args = parser.parse_args()
    
    train(args.dataset_path, args.output_dir, args.epochs, args.batch_size)
