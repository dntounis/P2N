"""P2N dataset + collator (multi-task).

Each example becomes the decoder sequence

    <s_task> {compact JSON target} </s_task> </s>

and we train with explicit teacher forcing:

    decoder_input_ids = seq[:-1]      (starts with the task prompt)
    labels            = seq[1:]       (padding -> -100)

This matches inference exactly, where generation is prompted with the
task start token.  (Previously ``VisionEncoderDecoderModel`` shifted the
labels itself and always prepended ``decoder_start_token_id = <s_p2n>``,
so auxiliary tasks were trained with a different prefix than the one used
at inference.)

Supported tasks:
  - p2n (main): full plot-to-numbers extraction
  - axis_info: axis scales, labels and orientation
  - element_count: legend / grid / error bars / colorbar / #series
"""
from __future__ import annotations

import json
import os
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image

from p2n_format import (  # noqa: F401  (re-exported for backwards compatibility)
    ALL_SPECIAL_TOKENS,
    TASK_TOKENS,
    build_target_text,
    load_metadata,
)

P2N_CONFIG_NAME = "p2n_config.json"


# ---------------------------------------------------------------------------
# Processor helpers
# ---------------------------------------------------------------------------
def add_task_tokens(processor, model=None):
    """Register task tokens as special tokens (never split, skipped on decode)."""
    tok = processor.tokenizer
    missing = [t for t in ALL_SPECIAL_TOKENS if t not in tok.get_vocab()]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing},
                               replace_additional_special_tokens=False)
    if model is not None and model.decoder.get_input_embeddings().weight.shape[0] < len(tok):
        model.decoder.resize_token_embeddings(len(tok))
    return missing


def set_image_size(processor, model, height: int, width: int):
    """Configure the processor/encoder input resolution (Donut-base default 2560x1920)."""
    ip = processor.image_processor
    ip.size = {"height": int(height), "width": int(width)}
    if model is not None:
        model.config.encoder.image_size = [int(height), int(width)]
        if hasattr(model.encoder, "config"):
            model.encoder.config.image_size = [int(height), int(width)]


def task_token_ids(tokenizer, task: str) -> tuple[int, int]:
    start, end = TASK_TOKENS.get(task, TASK_TOKENS["p2n"])
    return tokenizer.convert_tokens_to_ids(start), tokenizer.convert_tokens_to_ids(end)


def encode_target(tokenizer, task: str, body: str) -> list[int]:
    start_id, end_id = task_token_ids(tokenizer, task)
    body_ids = tokenizer(body, add_special_tokens=False).input_ids
    return [start_id] + body_ids + [end_id, tokenizer.eos_token_id]


def save_p2n_config(output_dir: str, **cfg):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, P2N_CONFIG_NAME), "w") as f:
        json.dump(cfg, f, indent=2)


def load_p2n_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, P2N_CONFIG_NAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"target_format": "verbose"}  # models trained before p2n_config.json existed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_image(image_path: str, use_torchvision_decode: bool = False):
    if use_torchvision_decode and image_path.lower().endswith((".jpg", ".jpeg")):
        try:
            from torchvision.io import ImageReadMode, decode_jpeg, read_file

            image = decode_jpeg(read_file(image_path), mode=ImageReadMode.RGB)
            return image.permute(1, 2, 0).numpy()
        except Exception:
            pass
    return Image.open(image_path).convert("RGB")


class P2NDataset(Dataset):
    """Reads ``metadata.jsonl`` and preprocesses examples on demand.

    Targets are tokenised once up-front so that over-length examples can be
    dropped (a truncated target never contains the end token, which teaches
    the model to never stop) and length statistics can be reported.
    """

    def __init__(
        self,
        dataset_path: str,
        processor,
        split: str | None = "train",
        max_length: int = 1024,
        tasks=None,
        plot_types=None,
        target_format: str = "compact",
        drop_overlength: bool = True,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        use_torchvision_decode: bool = False,
        max_samples: int | None = None,
        verbose: bool = True,
    ):
        self.processor = processor
        self.max_length = max_length
        self.use_torchvision_decode = use_torchvision_decode
        rows = load_metadata(dataset_path, split=split, tasks=tasks, plot_types=plot_types,
                             val_fraction=val_fraction, test_fraction=test_fraction)
        if max_samples:
            rows = rows[:max_samples]

        tok = processor.tokenizer
        self.items: list[tuple[str, list[int]]] = []
        self.rows: list[dict] = []
        lengths, dropped, truncated = [], Counter(), 0
        bodies = [build_target_text(r["ground_truth"], target_format=target_format) for r in rows]
        # Batch-tokenise bodies for speed.
        enc = tok([b for _, b in bodies], add_special_tokens=False).input_ids if bodies else []
        for r, (task, _), body_ids in zip(rows, bodies, enc):
            start_id, end_id = task_token_ids(tok, task)
            ids = [start_id] + body_ids + [end_id, tok.eos_token_id]
            lengths.append(len(ids))
            if len(ids) > max_length + 1:  # +1: decoder sees seq[:-1]
                if drop_overlength:
                    dropped[r["file_name"].split("/")[1] if "/" in r["file_name"] else "?"] += 1
                    continue
                ids = ids[: max_length + 1]
                truncated += 1
            self.items.append((r["image_path"], ids))
            self.rows.append(r)

        if verbose and lengths:
            ls = sorted(lengths)
            pct = lambda q: ls[min(len(ls) - 1, int(q * len(ls)))]  # noqa: E731
            print(f"[P2NDataset] split={split} rows={len(rows)} kept={len(self.items)} "
                  f"len p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={ls[-1]} "
                  f"(max_length={max_length})")
            if dropped:
                top = ", ".join(f"{k}:{v}" for k, v in dropped.most_common(8))
                print(f"[P2NDataset] dropped {sum(dropped.values())} over-length targets ({top})")
            if truncated:
                print(f"[P2NDataset] truncated {truncated} over-length targets")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, ids = self.items[idx]
        image = load_image(image_path, self.use_torchvision_decode)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values[0]
        return {"pixel_values": pixel_values, "input_ids": ids}


class P2NCollator:
    """Dynamic padding + teacher-forcing shift."""

    def __init__(self, pad_token_id: int, ignore_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_id = ignore_id

    def __call__(self, batch):
        pixel_values = torch.stack([torch.as_tensor(b["pixel_values"]) for b in batch]).float()
        seqs = [list(b["input_ids"]) for b in batch]
        n = max(len(s) for s in seqs) - 1
        dec = torch.full((len(seqs), n), self.pad_token_id, dtype=torch.long)
        lab = torch.full((len(seqs), n), self.ignore_id, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            k = len(s) - 1
            dec[i, :k] = torch.tensor(s[:-1])
            lab[i, :k] = torch.tensor(s[1:])
            att[i, :k] = 1
        return {"pixel_values": pixel_values, "decoder_input_ids": dec,
                "decoder_attention_mask": att, "labels": lab}


# Backwards-compatible alias
P2NMetadataDataset = P2NDataset


def collate_fn(batch, pad_token_id: int = 1):
    return P2NCollator(pad_token_id)(batch)
