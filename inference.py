"""P2N Inference — run a trained model on plot images.

Supports all three tasks:
  - p2n: Extract full structured data from a plot
  - axis_info: Detect axis scales, labels, orientation
  - element_count: Detect structural elements (legend, grid, error bars, etc.)

Usage:
    # Single image, main task
    python inference.py --model_dir ./p2n-model --image plot.png

    # Single image, specific task
    python inference.py --model_dir ./p2n-model --image plot.png --task axis_info

    # Batch mode on a directory
    python inference.py --model_dir ./p2n-model --image_dir ./test_images/ --task p2n --batch_size 8

    # All tasks on one image
    python inference.py --model_dir ./p2n-model --image plot.png --task all
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from dataset import add_task_tokens, load_p2n_config, task_token_ids
from p2n_format import TASK_TOKENS, expand_p2n, parse_prediction, strip_task_tokens


def load_model(model_dir, device=None, dtype=None):
    """Load trained model + processor.  The P2N config is attached as ``model.p2n_config``."""
    try:
        processor = DonutProcessor.from_pretrained(model_dir)
    except (OSError, ValueError):
        # Trainer checkpoints (run/checkpoint-N) don't contain the processor; use the run dir's.
        processor = DonutProcessor.from_pretrained(os.path.dirname(os.path.normpath(model_dir)))
    add_task_tokens(processor)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    if dtype is not None:
        model = model.to(dtype)
    model.to(device)
    model.eval()
    cfg_dir = model_dir if os.path.exists(os.path.join(model_dir, "p2n_config.json")) \
        else os.path.dirname(os.path.normpath(model_dir))
    model.p2n_config = load_p2n_config(cfg_dir) if os.path.isdir(cfg_dir) else {}
    print(f"Model loaded on {device}.")
    return model, processor, device


def _autocast(device):
    if str(device).startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def generate_texts(model, processor, images, task="p2n", max_new_tokens=1024, num_beams=1):
    """Batched greedy/beam generation for one task.  Returns decoded strings."""
    tok = processor.tokenizer
    images = [im.convert("RGB") if isinstance(im, Image.Image) else im for im in images]
    pixel_values = processor(images, return_tensors="pt").pixel_values
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device=device, dtype=next(model.parameters()).dtype)

    start_id, end_id = task_token_ids(tok, task)
    prompt = torch.full((len(images), 1), start_id, dtype=torch.long, device=device)
    eos_ids = [i for i in {end_id, tok.eos_token_id} if i is not None and i != tok.unk_token_id]

    with _autocast(device):
        out = model.generate(
            pixel_values,
            decoder_input_ids=prompt,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_ids,
            use_cache=True,
            num_beams=num_beams,
            do_sample=False,
            bad_words_ids=[[tok.unk_token_id]],
        )
    texts = tok.batch_decode(out, skip_special_tokens=True)
    # Older checkpoints registered task tokens as normal tokens; strip explicitly.
    return [strip_task_tokens(t) for t in texts]


def postprocess(text, task, target_format="compact"):
    obj, status = parse_prediction(text)
    if obj is None:
        return {"raw_output": text, "parse_error": True}
    if task == "p2n" and isinstance(obj, dict):
        obj = expand_p2n(obj)
    if status == "repaired" and isinstance(obj, dict):
        obj["_repaired_truncated_output"] = True
    return obj


def infer(model, processor, device, image_path, task="p2n", max_length=1024, num_beams=1):
    """Run inference on a single image for one task; returns parsed JSON (or raw text)."""
    image = Image.open(image_path).convert("RGB")
    text = generate_texts(model, processor, [image], task, max_length, num_beams)[0]
    return postprocess(text, task)


def run_all_tasks(model, processor, device, image_path, max_length=1024):
    """Run all three tasks on a single image."""
    return {task: infer(model, processor, device, image_path, task, max_length) for task in TASK_TOKENS}


def main():
    parser = argparse.ArgumentParser(description="Run P2N inference on plot images.")
    parser.add_argument("--model_dir", type=str, default="./p2n-model", help="Path to trained model")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to a directory of images")
    parser.add_argument("--task", type=str, default="p2n",
                        choices=["p2n", "axis_info", "element_count", "all"], help="Task to run")
    parser.add_argument("--max_length", type=int, default=1024, help="Max new decoder tokens")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Must specify either --image or --image_dir")

    model, processor, device = load_model(args.model_dir)

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            glob.glob(os.path.join(args.image_dir, "*.png")) +
            glob.glob(os.path.join(args.image_dir, "*.jpg")) +
            glob.glob(os.path.join(args.image_dir, "*.jpeg"))
        )
    print(f"Processing {len(image_paths)} image(s) with task='{args.task}'")

    tasks = list(TASK_TOKENS) if args.task == "all" else [args.task]
    all_results = {os.path.basename(p): {} for p in image_paths}
    for task in tasks:
        for i in range(0, len(image_paths), args.batch_size):
            chunk = image_paths[i:i + args.batch_size]
            texts = generate_texts(model, processor, [Image.open(p) for p in chunk], task,
                                   args.max_length, args.num_beams)
            for p, t in zip(chunk, texts):
                all_results[os.path.basename(p)][task] = postprocess(t, task)

    if args.task != "all":
        all_results = {k: v[args.task] for k, v in all_results.items()}
    for name, res in all_results.items():
        print(f"\n{'─' * 50}\nImage: {name}")
        print(json.dumps(res, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
