"""Shared, dependency-light helpers for the P2N data format.

This module is the single source of truth for:
  * task prompt tokens (``<s_p2n>`` etc.)
  * turning a ``metadata.jsonl`` ground-truth string into the decoder target text
  * robust parsing of (possibly truncated) model output back into JSON
  * deterministic, image-level train/val/test splitting

It deliberately imports only the standard library so it can be used from the
data generator, the trainer, the evaluator and the unit tests alike.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Task tokens
# ---------------------------------------------------------------------------
TASK_TOKENS = {
    "p2n": ("<s_p2n>", "</s_p2n>"),
    "axis_info": ("<s_axis_info>", "</s_axis_info>"),
    "element_count": ("<s_element_count>", "</s_element_count>"),
}
ALL_SPECIAL_TOKENS = [tok for pair in TASK_TOKENS.values() for tok in pair]

# Fields that describe how the *image file* was produced rather than what the
# plot shows.  They cannot be read from the pixels in a meaningful way (or are
# trivially redundant) and only waste decoder tokens, so they are stripped from
# training targets.
NON_VISUAL_P2N_FIELDS = ("degradation_effects", "image_quality")

DEFAULT_SIG_FIGS = 4


# ---------------------------------------------------------------------------
# Ground-truth handling
# ---------------------------------------------------------------------------
def parse_ground_truth(gt_text: str | dict) -> tuple[str, dict]:
    """Return ``(task, gt_parse)`` for a metadata ``ground_truth`` entry.

    Accepts both the wrapped form ``{"gt_parse": {...}, "task": "p2n"}`` and a
    bare dict (treated as p2n).
    """
    if isinstance(gt_text, dict):
        gt = gt_text
    else:
        try:
            gt = json.loads(gt_text)
        except (json.JSONDecodeError, TypeError):
            return "p2n", {}
    if not isinstance(gt, dict):
        return "p2n", {}
    task = gt.get("task", "p2n")
    if task not in TASK_TOKENS:
        task = "p2n"
    gt_parse = gt.get("gt_parse", gt)
    return task, gt_parse if isinstance(gt_parse, dict) else {}


def round_sig(x: float, sig: int = DEFAULT_SIG_FIGS) -> float | int:
    """Round to ``sig`` significant figures (keeps ints as ints)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if not math.isfinite(x):
        return x
    if x == 0:
        return 0.0
    digits = sig - int(math.floor(math.log10(abs(x)))) - 1
    r = round(x, digits)
    # Avoid "-0.0" and float noise like 0.30000000000000004
    return float(f"{r:.{sig}g}") if r != 0 else 0.0


def normalize_values(obj: Any, sig: int = DEFAULT_SIG_FIGS) -> Any:
    """Recursively convert numpy scalars, round floats and drop NaN/inf."""
    # numpy scalars expose .item(); avoid importing numpy here.
    if hasattr(obj, "item") and not isinstance(obj, (list, dict, str)):
        try:
            obj = obj.item()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round_sig(obj, sig)
    if isinstance(obj, dict):
        return {str(k): normalize_values(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_values(v, sig) for v in obj]
    if hasattr(obj, "tolist"):
        return normalize_values(obj.tolist(), sig)
    return obj


# ---------------------------------------------------------------------------
# Compact (columnar) data_series encoding
# ---------------------------------------------------------------------------
# A typical scatter target repeats every key for every point:
#   [{"type":"scatter","x":1.2,"y":3.4,"x_log":false,"y_log":false}, ...]
# The compact form groups consecutive records that share a type and key set,
# hoists fields that are constant within the group and stores the rest as
# rows of a table:
#   [{"type":"scatter","x_log":false,"y_log":false,"cols":["x","y"],
#     "rows":[[1.2,3.4], ...]}]
# This is lossless (see ``expand_series``) and typically 2-3x shorter in
# decoder tokens.
COLS_KEY, ROWS_KEY = "cols", "rows"


def compact_series(records: list) -> list:
    if not isinstance(records, list):
        return records
    groups: list[list[dict]] = []
    for rec in records:
        if (isinstance(rec, dict) and groups and isinstance(groups[-1][0], dict)
                and rec.get("type") == groups[-1][0].get("type")
                and list(rec.keys()) == list(groups[-1][0].keys())
                and COLS_KEY not in rec and ROWS_KEY not in rec):
            groups[-1].append(rec)
        else:
            groups.append([rec])
    out = []
    for g in groups:
        if len(g) < 2 or not isinstance(g[0], dict):
            out.extend(g)
            continue
        keys = list(g[0].keys())
        const = [k for k in keys if all(r[k] == g[0][k] for r in g[1:])]
        cols = [k for k in keys if k not in const]
        if not cols:
            out.extend(g)  # fully identical records; keep verbatim
            continue
        block = {k: g[0][k] for k in const}
        block[COLS_KEY] = cols
        block[ROWS_KEY] = [[r[k] for k in cols] for r in g]
        out.append(block)
    return out


def expand_series(series: list) -> list:
    """Inverse of :func:`compact_series`; tolerant of malformed model output."""
    if not isinstance(series, list):
        return []
    out = []
    for item in series:
        if isinstance(item, dict) and COLS_KEY in item and ROWS_KEY in item:
            cols, rows = item.get(COLS_KEY), item.get(ROWS_KEY)
            if not isinstance(cols, list) or not isinstance(rows, list):
                continue
            const = {k: v for k, v in item.items() if k not in (COLS_KEY, ROWS_KEY)}
            for row in rows:
                if not isinstance(row, list):
                    continue
                rec = dict(const)
                rec.update({str(c): v for c, v in zip(cols, row)})
                out.append(rec)
        else:
            out.append(item)
    return out


def _map_panels(obj: dict, fn) -> dict:
    if not isinstance(obj, dict) or not isinstance(obj.get("panels"), list):
        return obj
    obj = dict(obj)
    new_panels = []
    for p in obj["panels"]:
        if isinstance(p, dict) and "data_series" in p:
            p = dict(p)
            p["data_series"] = fn(p["data_series"])
        new_panels.append(p)
    obj["panels"] = new_panels
    return obj


def compact_p2n(obj: dict) -> dict:
    return _map_panels(obj, compact_series)


def expand_p2n(obj: dict) -> dict:
    return _map_panels(obj, expand_series)


TARGET_FORMATS = ("compact", "verbose")


def canonical_target(task: str, gt_parse: dict, sig: int = DEFAULT_SIG_FIGS,
                     target_format: str = "compact") -> dict:
    """The dict the model is trained to emit for ``task``."""
    if task == "p2n":
        gt_parse = {k: v for k, v in gt_parse.items() if k not in NON_VISUAL_P2N_FIELDS}
        gt_parse = normalize_values(gt_parse, sig)
        return compact_p2n(gt_parse) if target_format == "compact" else gt_parse
    if task in ("axis_info", "element_count"):
        gt_parse = {k: v for k, v in gt_parse.items() if k != "task_type"}
    return normalize_values(gt_parse, sig)


def to_json(obj: Any) -> str:
    """Compact JSON serialisation used for decoder targets."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def build_target_text(gt_text: str | dict, sig: int = DEFAULT_SIG_FIGS,
                      target_format: str = "compact") -> tuple[str, str]:
    """Return ``(task, body)`` where ``body`` is the JSON training target.

    The decoder sequence is ``<s_task> body </s_task> </s>``; the prompt and
    terminator tokens are added by the dataset so the body can be reused for
    evaluation.
    """
    task, gt_parse = parse_ground_truth(gt_text)
    return task, to_json(canonical_target(task, gt_parse, sig, target_format))


# ---------------------------------------------------------------------------
# Parsing model output
# ---------------------------------------------------------------------------
_TRAILING_PARTIAL = re.compile(r'[,:]\s*$')


def strip_task_tokens(text: str, extra_tokens: Iterable[str] = ()) -> str:
    for tok in list(ALL_SPECIAL_TOKENS) + list(extra_tokens):
        if tok:
            text = text.replace(tok, "")
    return text.strip()


def repair_json(text: str) -> str | None:
    """Best-effort completion of a truncated JSON document.

    Walks the string tracking open brackets/strings, cuts back to the last
    position where a complete value ended, and appends the missing closers.
    Returns ``None`` if nothing sensible can be recovered.
    """
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    text = text[start:]

    stack: list[str] = []
    in_str = False
    esc = False
    # (index_after_char, stack_snapshot) at points where truncation is safe
    safe_cut: tuple[int, list[str]] | None = None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                break  # structurally broken; stop here
            stack.pop()
            safe_cut = (i + 1, list(stack))
            if not stack:
                return text[: i + 1]
        elif ch == ",":
            safe_cut = (i, list(stack))
    if safe_cut is None:
        return None
    cut, open_stack = safe_cut
    body = _TRAILING_PARTIAL.sub("", text[:cut].rstrip())
    return body + "".join(reversed(open_stack))


def parse_prediction(text: str) -> tuple[Any, str]:
    """Parse model output.  Returns ``(obj, status)``.

    ``status`` is one of ``"valid"`` (parsed as-is), ``"repaired"`` (parsed
    after closing truncated brackets) or ``"invalid"`` (``obj`` is ``None``).
    """
    text = strip_task_tokens(text)
    try:
        return json.loads(text), "valid"
    except (json.JSONDecodeError, TypeError):
        pass
    fixed = repair_json(text)
    if fixed is not None:
        try:
            return json.loads(fixed), "repaired"
        except json.JSONDecodeError:
            pass
    return None, "invalid"


# ---------------------------------------------------------------------------
# Deterministic splits
# ---------------------------------------------------------------------------
def split_of(file_name: str, val_fraction: float = 0.05, test_fraction: float = 0.05) -> str:
    """Assign an *image* to ``train``/``val``/``test`` by hashing its path.

    Hashing the image path (not the metadata row) guarantees that the
    auxiliary-task rows of an image land in the same split as its main row,
    so no image leaks between train and evaluation.
    """
    h = int(hashlib.md5(file_name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_fraction:
        return "test"
    if h < test_fraction + val_fraction:
        return "val"
    return "train"


def load_metadata(dataset_path: str, split: str | None = None, tasks: Iterable[str] | None = None,
                  val_fraction: float = 0.05, test_fraction: float = 0.05,
                  plot_types: Iterable[str] | None = None) -> list[dict]:
    """Read ``metadata.jsonl`` into a list of ``{file_name, image_path, ground_truth, task}``.

    Optional filters: ``split`` (train/val/test), ``tasks`` and ``plot_types``
    (matched against the ``images/<plot_type>/`` directory).
    """
    import os

    metadata_path = os.path.join(dataset_path, "metadata.jsonl")
    tasks = set(tasks) if tasks else None
    plot_types = set(plot_types) if plot_types else None
    rows = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            fn = entry["file_name"]
            if split and split_of(fn, val_fraction, test_fraction) != split:
                continue
            task, _ = parse_ground_truth(entry["ground_truth"])
            if tasks and task not in tasks:
                continue
            if plot_types:
                parts = fn.split("/")
                if len(parts) < 3 or parts[1] not in plot_types:
                    continue
            path = os.path.join(dataset_path, fn)
            if not os.path.exists(path):
                continue
            rows.append({"file_name": fn, "image_path": path,
                         "ground_truth": entry["ground_truth"], "task": task})
    return rows
