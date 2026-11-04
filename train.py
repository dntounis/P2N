"""Fine-tune Donut (naver-clova-ix/donut-base) for plot-to-numbers extraction.

Single GPU:
    python train.py --dataset_path data --output_dir p2n-model

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train.py --dataset_path data --output_dir p2n-model \
        --batch_size 4 --gradient_accumulation_steps 4

Notes
  * Splits are image-level and deterministic (``p2n_format.split_of``):
    train / val (used for eval_loss + generation metrics) / test (held out
    for ``evaluate.py``).
  * Targets use the compact columnar JSON format by default
    (``--target_format compact``); over-length targets are dropped rather
    than truncated.
  * ``--gen_eval_samples N`` decodes N validation images at each evaluation
    and logs value_acc@5 / nMAE / parse rate — the metrics we actually care
    about — alongside eval_loss.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import (
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    VisionEncoderDecoderModel,
)

from dataset import (
    P2NCollator,
    P2NDataset,
    add_task_tokens,
    save_p2n_config,
    set_image_size,
)
from p2n_format import TASK_TOKENS, TARGET_FORMATS

DEFAULT_MODEL_ID = "naver-clova-ix/donut-base"


def token_cross_entropy(logits, labels, num_items_in_batch=None):
    """Next-token CE where ``labels[t]`` is the target for ``logits[t]`` (no extra shift).

    transformers >= 4.46 routes VisionEncoderDecoderModel's loss through
    ``ForCausalLMLoss``, which shifts labels by one more position.  With
    explicit ``decoder_input_ids`` (or the classic shift-right setup) that
    trains the model to predict the token *after* next: low loss, garbage
    generation.  Computing the loss here keeps training correct across
    transformers versions.
    """
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1).to(logits.device),
        ignore_index=-100, reduction="sum")
    if num_items_in_batch is None:
        num_items_in_batch = (labels != -100).sum()
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(loss.device)
    return loss / num_items_in_batch


class P2NTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with a version-independent token cross-entropy loss."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Mirror Trainer's normalisation contract: when the model "accepts loss kwargs"
        # Trainer expects a loss already normalised by the token count of the whole
        # accumulated batch; otherwise it divides a per-batch mean by the GA steps.
        token_normalised = getattr(self, "model_accepts_loss_kwargs", False) and num_items_in_batch is not None
        loss = token_cross_entropy(outputs.logits, labels, num_items_in_batch if token_normalised else None)
        if token_normalised and getattr(self.args, "average_tokens_across_devices", False):
            # Trainer later divides by the number of processes (as in Trainer.compute_loss)
            loss = loss * (self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu)
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


class GenerationEvalCallback(TrainerCallback):
    """Decode a fixed set of validation samples and log task metrics (rank 0 only)."""

    def __init__(self, dataset: P2NDataset, processor, max_new_tokens: int, batch_size: int,
                 log_path: str | None = None):
        self.dataset = dataset
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.log_path = log_path
        self.trainer = None

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not state.is_world_process_zero or model is None or len(self.dataset) == 0:
            return
        from collections import defaultdict

        from PIL import Image

        from evaluate import score_rows, summarize
        from inference import generate_texts

        was_training = model.training
        model.eval()
        t0 = time.time()
        rows = self.dataset.rows
        texts = [None] * len(rows)
        by_task = defaultdict(list)
        for i, r in enumerate(rows):
            by_task[r["task"]].append(i)
        for task, idxs in by_task.items():
            for b in range(0, len(idxs), self.batch_size):
                chunk = idxs[b:b + self.batch_size]
                images = [Image.open(rows[i]["image_path"]).convert("RGB") for i in chunk]
                for i, t in zip(chunk, generate_texts(model, self.processor, images, task,
                                                      self.max_new_tokens)):
                    texts[i] = t
        report = summarize(score_rows(rows, texts))
        if was_training:
            model.train()

        flat = {}
        p2n = report.get("p2n", {})
        for k in ("json_parse_rate", "value_acc@5", "value_acc@1", "nmae", "record_f1",
                  "count_exact_acc", "axes_range_err"):
            if p2n.get(k) is not None:
                flat[f"gen_{k}"] = round(p2n[k], 4)
        for task in ("axis_info", "element_count"):
            for k, v in report.get(task, {}).items():
                if k.endswith("_acc") and k in ("all_correct_acc", "has_legend_acc", "plot_type_acc",
                                                "x_scale_acc", "y_scale_acc"):
                    flat[f"gen_{task}_{k}"] = round(v, 4)
        flat["gen_seconds"] = round(time.time() - t0, 1)
        print(f"\n[gen-eval step {state.global_step}] " + json.dumps(flat))
        if metrics is not None:
            metrics.update({f"eval_{k}": v for k, v in flat.items()})
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **flat}) + "\n")
        # Show one example to make failure modes visible in logs.
        if texts:
            print(f"[gen-eval sample] {rows[0]['file_name']}: {texts[0][:300]}")


def build_model_and_processor(model_id, image_size, max_length, processor_dir=None):
    processor = DonutProcessor.from_pretrained(processor_dir or model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    add_task_tokens(processor, model)
    set_image_size(processor, model, *image_size)
    # Donut's sequence-level augmentations aren't needed: plots are rendered upright.
    processor.image_processor.do_align_long_axis = False

    tok = processor.tokenizer
    model.config.pad_token_id = tok.pad_token_id
    model.config.decoder_start_token_id = tok.convert_tokens_to_ids(TASK_TOKENS["p2n"][0])
    model.config.eos_token_id = tok.eos_token_id
    model.config.decoder.max_length = max_length
    if getattr(model.config.decoder, "max_position_embeddings", None) and \
            model.config.decoder.max_position_embeddings < max_length + 2:
        raise ValueError(f"max_length={max_length} exceeds decoder max_position_embeddings="
                         f"{model.config.decoder.max_position_embeddings}")
    return model, processor


def train(args):
    model, processor = build_model_and_processor(args.model_id, args.image_size, args.max_length)

    common = dict(
        max_length=args.max_length,
        tasks=args.tasks,
        plot_types=args.plot_types,
        target_format=args.target_format,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        use_torchvision_decode=args.torchvision_decode,
    )
    train_dataset = P2NDataset(args.dataset_path, processor, split="train",
                               max_samples=args.max_train_samples, **common)
    eval_dataset = P2NDataset(args.dataset_path, processor, split="val",
                              max_samples=args.max_eval_samples, **common)
    if len(eval_dataset) == 0:
        eval_dataset = None
    gen_dataset = None
    if args.gen_eval_samples and eval_dataset is not None:
        gen_dataset = P2NDataset(args.dataset_path, processor, split="val",
                                 max_samples=args.gen_eval_samples,
                                 **{**common, "tasks": args.gen_eval_tasks}, verbose=False)

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported() and not args.fp16
    fp16 = use_cuda and not bf16 and not args.fp32
    if args.fp32:
        bf16 = fp16 = False

    if eval_dataset is not None and args.save_steps % args.eval_steps:
        # load_best_model_at_end needs save_steps to be a multiple of eval_steps
        args.save_steps = -(-args.save_steps // args.eval_steps) * args.eval_steps
        print(f"Rounded save_steps up to {args.save_steps} (multiple of eval_steps)")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_pin_memory=use_cuda,
        group_by_length=False,
        report_to=args.report_to,
        seed=args.seed,
        ddp_find_unused_parameters=False,
        use_cpu=not use_cuda,
    )

    callbacks = []
    if gen_dataset is not None and len(gen_dataset):
        callbacks.append(GenerationEvalCallback(
            gen_dataset, processor, args.max_length, args.batch_size,
            log_path=os.path.join(args.output_dir, "gen_eval_log.jsonl")))

    os.makedirs(args.output_dir, exist_ok=True)
    p2n_cfg = dict(target_format=args.target_format, image_size=list(args.image_size),
                   max_length=args.max_length, base_model=args.model_id,
                   tasks=args.tasks or list(TASK_TOKENS), plot_types=args.plot_types,
                   val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    save_p2n_config(args.output_dir, **p2n_cfg)
    processor.save_pretrained(args.output_dir)

    trainer = P2NTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=P2NCollator(processor.tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    save_p2n_config(args.output_dir, **p2n_cfg)
    print(f"Model saved to {args.output_dir}")


def build_parser():
    p = argparse.ArgumentParser(description="Train P2N model.")
    p.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset")
    p.add_argument("--output_dir", type=str, default="./p2n-model", help="Output directory for model")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                   help="Base checkpoint (default donut-base; a local path is useful for smoke tests)")
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"),
                   help="Encoder input resolution (donut-base native: 2560 1920)")
    p.add_argument("--target_format", type=str, default="compact", choices=TARGET_FORMATS)
    p.add_argument("--tasks", type=str, nargs="+", default=None, choices=list(TASK_TOKENS),
                   help="Tasks to train on (default: all present in the data)")
    p.add_argument("--plot_types", type=str, nargs="+", default=None,
                   help="Restrict training to these plot types (curriculum / ablations)")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--test_fraction", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=30, help="Number of training epochs")
    p.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs)")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    p.add_argument("--learning_rate", type=float, default=3e-5, help="Peak learning rate")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_length", type=int, default=1024, help="Max decoder sequence length")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per process")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=500)
    p.add_argument("--gen_eval_samples", type=int, default=64,
                   help="Validation samples decoded at each eval for task metrics (0 disables)")
    p.add_argument("--gen_eval_tasks", type=str, nargs="+", default=["p2n"], choices=list(TASK_TOKENS))
    p.add_argument("--fp16", action="store_true", help="Force fp16 instead of bf16 on CUDA")
    p.add_argument("--fp32", action="store_true", help="Disable mixed precision")
    p.add_argument("--torchvision_decode", action="store_true",
                   help="Decode JPEGs with torchvision instead of PIL")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
