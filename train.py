import os
import argparse
import json
import torch
from torch.utils.data import random_split
from transformers import VisionEncoderDecoderModel, DonutProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from dataset import P2NMetadataDataset, load_p2n_imagefolder

def train(
    dataset_path,
    output_dir,
    epochs,
    max_steps,
    batch_size,
    learning_rate,
    max_length,
    gradient_accumulation_steps,
    preprocessed_path=None,
    on_the_fly=False,
    torchvision_decode=False,
    cuda_jpeg_decode=False,
):
    # Load model and processor
    model_id = "naver-clova-ix/donut-base"
    if preprocessed_path:
        processor = DonutProcessor.from_pretrained(os.path.join(preprocessed_path, "processor"))
    else:
        processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Add special tokens for main task and auxiliary sub-tasks
    new_tokens = [
        "<s_p2n>", "</s_p2n>",
        "<s_axis_info>", "</s_axis_info>",
        "<s_element_count>", "</s_element_count>",
    ]
    processor.tokenizer.add_tokens(new_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Model configuration for generation
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_p2n>"])[0]

    if preprocessed_path:
        print(f"Loading preprocessed dataset from {preprocessed_path}...")
        train_dataset = load_from_disk(os.path.join(preprocessed_path, "train"))
        eval_path = os.path.join(preprocessed_path, "eval")
        eval_dataset = load_from_disk(eval_path) if os.path.exists(eval_path) else None
    elif on_the_fly:
        full_dataset = P2NMetadataDataset(
            dataset_path,
            processor,
            max_length=max_length,
            use_torchvision_decode=torchvision_decode,
            cuda_jpeg_decode=cuda_jpeg_decode,
        )
        eval_size = max(1, int(len(full_dataset) * 0.1)) if len(full_dataset) > 1 else 0
        train_size = len(full_dataset) - eval_size
        if eval_size:
            train_dataset, eval_dataset = random_split(
                full_dataset,
                [train_size, eval_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            train_dataset = full_dataset
            eval_dataset = None
    else:
        dataset = load_p2n_imagefolder(dataset_path)

        if "train" in dataset:
            split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        def preprocess(example):
            image = example["image"].convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()

            gt_text = example.get("ground_truth", "{}")
            try:
                gt_parsed = json.loads(gt_text)
                task = gt_parsed.get("task", "p2n")
            except (json.JSONDecodeError, TypeError):
                task = "p2n"

            task_tokens = {
                "p2n": ("<s_p2n>", "</s_p2n>"),
                "axis_info": ("<s_axis_info>", "</s_axis_info>"),
                "element_count": ("<s_element_count>", "</s_element_count>"),
            }
            start_tok, end_tok = task_tokens.get(task, ("<s_p2n>", "</s_p2n>"))
            target_text = f"{start_tok}{gt_text}{end_tok}"

            labels = processor.tokenizer(
                target_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()

            labels[labels == processor.tokenizer.pad_token_id] = -100

            return {"pixel_values": pixel_values, "labels": labels}

        train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names,
                                          writer_batch_size=50)
        if eval_dataset:
            eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names,
                                            writer_batch_size=50)

    if hasattr(train_dataset, "set_format"):
        train_dataset.set_format("torch")
    if eval_dataset and hasattr(eval_dataset, "set_format"):
        eval_dataset.set_format("torch")

    # Training arguments — tuned for 50K dataset scale
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True,
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
        dataloader_num_workers=0,
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
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (useful for smoke tests)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max decoder sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--preprocessed_path", type=str, default=None,
                        help="Path to preprocessed dataset (from preprocess_data.py). Skips on-the-fly preprocessing.")
    parser.add_argument("--on_the_fly", action="store_true",
                        help="Preprocess directly in the training dataloader instead of building an Arrow cache.")
    parser.add_argument("--torchvision_decode", action="store_true",
                        help="Use torchvision image decoding for JPEG inputs when --on_the_fly is set.")
    parser.add_argument("--cuda_jpeg_decode", action="store_true",
                        help="Use torchvision CUDA JPEG decode when --on_the_fly and --torchvision_decode are set.")
    args = parser.parse_args()

    train(args.dataset_path, args.output_dir, args.epochs, args.max_steps, args.batch_size,
          args.learning_rate, args.max_length, args.gradient_accumulation_steps, args.preprocessed_path,
          args.on_the_fly, args.torchvision_decode, args.cuda_jpeg_decode)
