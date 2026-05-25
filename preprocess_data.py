import os
import argparse
import json
from transformers import DonutProcessor
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for P2N training (run on CPU node with many cores)")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to raw imagefolder dataset")
    parser.add_argument("--output_dir", type=str, default="data_preprocessed", help="Where to save preprocessed Arrow dataset")
    parser.add_argument("--max_length", type=int, default=768, help="Max decoder sequence length")
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Number of parallel workers (default: auto-detect from OS affinity)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction for eval split")
    args = parser.parse_args()

    num_proc = args.num_proc if args.num_proc else len(os.sched_getaffinity(0))
    print(f"Preprocessing with {num_proc} CPU workers")

    model_id = "naver-clova-ix/donut-base"
    processor = DonutProcessor.from_pretrained(model_id)

    new_tokens = [
        "<s_p2n>", "</s_p2n>",
        "<s_axis_info>", "</s_axis_info>",
        "<s_element_count>", "</s_element_count>",
    ]
    processor.tokenizer.add_tokens(new_tokens)

    max_length = args.max_length

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

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset("imagefolder", data_dir=args.dataset_path)

    if "train" in dataset:
        split = dataset["train"].train_test_split(test_size=args.test_size, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Preprocessing {len(train_dataset)} training examples...")
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
        num_proc=num_proc,
        writer_batch_size=100,
        desc="Preprocessing train",
    )

    if eval_dataset:
        print(f"Preprocessing {len(eval_dataset)} eval examples...")
        eval_dataset = eval_dataset.map(
            preprocess,
            remove_columns=eval_dataset.column_names,
            num_proc=num_proc,
            writer_batch_size=100,
            desc="Preprocessing eval",
        )

    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(args.output_dir, "train"))
    if eval_dataset:
        eval_dataset.save_to_disk(os.path.join(args.output_dir, "eval"))

    processor.save_pretrained(os.path.join(args.output_dir, "processor"))
    print(f"Done! Saved preprocessed dataset to {args.output_dir}/")
    print(f"  Train: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"  Eval:  {len(eval_dataset)} examples")
    print(f"\nTo train, run:")
    print(f"  torchrun --nproc_per_node=4 train.py --preprocessed_path {args.output_dir} ...")


if __name__ == "__main__":
    main()
