import argparse
from transformers import VisionEncoderDecoderModel, DonutProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

def train(dataset_path, output_dir, epochs, batch_size, learning_rate, max_length, gradient_accumulation_steps):
    # Load model and processor
    model_id = "naver-clova-ix/donut-base"
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Add special token for our task
    new_tokens = ["<s_p2n>", "</s_p2n>"]
    processor.tokenizer.add_tokens(new_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Model configuration for generation
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_p2n>"])[0]
    model.config.max_length = max_length

    # Load dataset using HF imagefolder format
    dataset = load_dataset("imagefolder", data_dir=dataset_path)

    # Train/val split (90/10)
    if "train" in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    def preprocess(example):
        """Process image and ground truth into model inputs."""
        image = example["image"].convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Parse ground truth text 
        gt_text = example.get("ground_truth", "{}")
        target_text = f"<s_p2n>{gt_text}</s_p2n>"
        
        labels = processor.tokenizer(
            target_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

    train_dataset.set_format("torch")
    if eval_dataset:
        eval_dataset.set_format("torch")

    # Training arguments — tuned for 50K dataset scale
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_num_workers=4,
        report_to="none",
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1536, help="Max decoder sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    args = parser.parse_args()
    
    train(args.dataset_path, args.output_dir, args.epochs, args.batch_size, 
          args.learning_rate, args.max_length, args.gradient_accumulation_steps)
