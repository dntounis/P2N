"""P2N Evaluation — structure-aware numeric metrics for all tasks.

Evaluates on the held-out *test* split by default (image-level hash split,
see ``p2n_format.split_of``) so no training image is ever scored.

Main task (p2n) metrics, see ``p2n_metrics`` for definitions:
  json_parse_rate / json_valid_rate, value_acc@{1,2,5,10}, nmae,
  record precision/recall/F1, count_exact_acc, plot_type_acc,
  matched MAE/RMSE (data units), axes range error.
Auxiliary tasks: per-field accuracy (axis_info, element_count), n_series MAE.

Usage:
    # Evaluate a model (writes predictions + report into --out_dir)
    python evaluate.py --model_dir ./p2n-model --dataset_path data --num_samples 200

    # Re-score saved predictions without a model
    python evaluate.py --predictions runs/eval/predictions.jsonl --dataset_path data

    # Sanity-check the metrics with perturbed ground truth (no model needed)
    python evaluate.py --oracle_noise 0.02 --dataset_path data
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

from p2n_format import canonical_target, load_metadata, parse_ground_truth, parse_prediction, to_json
from p2n_metrics import (
    aggregate_fields,
    aggregate_p2n,
    score_axis_info,
    score_element_count,
    score_p2n,
)

TASKS = ("p2n", "axis_info", "element_count")


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------
def select_rows(dataset_path, split, tasks, num_samples, plot_types=None, seed=42,
                val_fraction=0.05, test_fraction=0.05, stratify=True):
    rows = load_metadata(dataset_path, split=split, tasks=tasks, plot_types=plot_types,
                         val_fraction=val_fraction, test_fraction=test_fraction)
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    out = []
    for task in tasks:
        cand = by_task.get(task, [])
        if num_samples and len(cand) > num_samples:
            if stratify:
                # Round-robin over plot types so rare types are represented.
                by_pt = defaultdict(list)
                for r in cand:
                    by_pt[r["file_name"].split("/")[1]].append(r)
                for v in by_pt.values():
                    rng.shuffle(v)
                keys = sorted(by_pt)
                rng.shuffle(keys)
                picked = []
                while len(picked) < num_samples:
                    for k in keys:
                        if by_pt[k] and len(picked) < num_samples:
                            picked.append(by_pt[k].pop())
                cand = picked
            else:
                cand = rng.sample(cand, num_samples)
        out.extend(cand)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_rows(rows, pred_texts):
    """Score predictions (raw decoded strings) against metadata rows."""
    per_task = defaultdict(lambda: {"scores": [], "statuses": [], "plot_types": []})
    for row, text in zip(rows, pred_texts):
        task, gt = parse_ground_truth(row["ground_truth"])
        pred, status = parse_prediction(text) if text is not None else (None, "invalid")
        if task == "p2n":
            s = score_p2n(pred, canonical_target("p2n", gt, target_format="verbose"))
        elif task == "axis_info":
            s = score_axis_info(pred, gt)
        else:
            s = score_element_count(pred, gt)
        d = per_task[task]
        d["scores"].append(s)
        d["statuses"].append(status)
        d["plot_types"].append(row["file_name"].split("/")[1])
    return per_task


def summarize(per_task):
    report = {}
    for task, d in per_task.items():
        if task == "p2n":
            summary = aggregate_p2n(d["scores"], d["statuses"])
            by_pt = defaultdict(lambda: ([], []))
            for s, st, pt in zip(d["scores"], d["statuses"], d["plot_types"]):
                by_pt[pt][0].append(s)
                by_pt[pt][1].append(st)
            summary["per_plot_type"] = {
                pt: {k: v for k, v in aggregate_p2n(sc, sts).items()
                     if k in ("n", "json_parse_rate", "value_acc@5", "nmae", "record_f1",
                              "count_exact_acc", "median_sample_mae", "axes_range_err")}
                for pt, (sc, sts) in sorted(by_pt.items())
            }
        else:
            summary = aggregate_fields(d["scores"])
            n = len(d["statuses"])
            summary["json_parse_rate"] = sum(s != "invalid" for s in d["statuses"]) / max(n, 1)
        report[task] = summary
    return report


def _fmt(v):
    if v is None:
        return "   -  "
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_report(report, top_k=15):
    if "p2n" in report:
        s = report["p2n"]
        print("=" * 72)
        print("TASK: Plot-to-Numbers (p2n)")
        print("=" * 72)
        for k in ("n", "json_parse_rate", "json_valid_rate", "value_acc@1", "value_acc@2",
                  "value_acc@5", "value_acc@10", "nmae", "record_precision", "record_recall",
                  "record_f1", "count_exact_acc", "plot_type_acc", "axes_range_err", "axes_scale_acc",
                  "median_sample_mae", "median_sample_rmse", "median_rel_error"):
            if k in s:
                print(f"  {k:22s} {_fmt(s[k])}")
        pts = s.get("per_plot_type", {})
        if pts:
            print(f"\n  Worst {min(top_k, len(pts))} plot types by value_acc@5:")
            print(f"    {'plot_type':28s} {'n':>4s} {'parse':>6s} {'acc@5':>6s} {'nmae':>6s} {'f1':>6s}")
            rows = sorted(pts.items(), key=lambda kv: (kv[1].get("value_acc@5") or 0.0))
            for pt, m in rows[:top_k]:
                print(f"    {pt:28s} {m['n']:4d} {m.get('json_parse_rate', 0):6.2f} "
                      f"{(m.get('value_acc@5') or 0):6.3f} {(m.get('nmae') or 0):6.3f} "
                      f"{(m.get('record_f1') or 0):6.3f}")
    for task in ("axis_info", "element_count"):
        if task in report:
            print(f"\n{'=' * 72}\nTASK: {task}\n{'=' * 72}")
            for k, v in report[task].items():
                print(f"  {k:22s} {_fmt(v)}")


# ---------------------------------------------------------------------------
# Prediction sources
# ---------------------------------------------------------------------------
def _perturb(obj, noise, rng):
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return obj * (1 + rng.gauss(0, noise)) if noise else obj
    if isinstance(obj, dict):
        return {k: _perturb(v, noise, rng) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_perturb(v, noise, rng) for v in obj]
    return obj


def oracle_predictions(rows, noise, drop_fraction=0.0, seed=0):
    """Ground truth with multiplicative noise / dropped records (metric sanity checks)."""
    rng = random.Random(seed)
    texts = []
    for r in rows:
        task, gt = parse_ground_truth(r["ground_truth"])
        tgt = canonical_target(task, gt, target_format="verbose")
        if task == "p2n" and drop_fraction:
            for p in tgt.get("panels", []):
                p["data_series"] = [x for x in p.get("data_series", []) if rng.random() >= drop_fraction]
        if task == "p2n":
            tgt = _perturb(tgt, noise, rng)
        texts.append(to_json(tgt))
    return texts


def model_predictions(rows, model_dir, batch_size, max_new_tokens, num_beams=1):
    from PIL import Image
    from tqdm import tqdm

    from inference import generate_texts, load_model

    model, processor, _ = load_model(model_dir)
    texts = [None] * len(rows)
    by_task = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[r["task"]].append(i)
    for task, idxs in by_task.items():
        for b in tqdm(range(0, len(idxs), batch_size), desc=f"generate[{task}]"):
            chunk = idxs[b:b + batch_size]
            images = [Image.open(rows[i]["image_path"]).convert("RGB") for i in chunk]
            outs = generate_texts(model, processor, images, task, max_new_tokens, num_beams)
            for i, t in zip(chunk, outs):
                texts[i] = t
    return texts


def evaluate(args):
    tasks = args.tasks or list(TASKS)
    if args.predictions:
        preds = {}
        with open(args.predictions) as f:
            for line in f:
                p = json.loads(line)
                preds[(p["file_name"], p["task"])] = p["prediction"]
        all_rows = load_metadata(args.dataset_path, split=None, tasks=tasks)
        rows = [r for r in all_rows if (r["file_name"], r["task"]) in preds]
        texts = [preds[(r["file_name"], r["task"])] for r in rows]
    else:
        rows = select_rows(args.dataset_path, args.split, tasks, args.num_samples, args.plot_types,
                           seed=args.seed, val_fraction=args.val_fraction,
                           test_fraction=args.test_fraction)
        if not rows:
            raise SystemExit(f"No samples found for split={args.split} tasks={tasks}")
        if args.oracle_noise is not None:
            texts = oracle_predictions(rows, args.oracle_noise, args.oracle_drop)
        else:
            texts = model_predictions(rows, args.model_dir, args.batch_size, args.max_length,
                                      args.num_beams)
    print(f"Scoring {len(rows)} predictions "
          f"({', '.join(f'{t}={sum(r['task'] == t for r in rows)}' for t in tasks)})\n")

    report = summarize(score_rows(rows, texts))
    print_report(report)

    out_dir = args.out_dir or (os.path.join(args.model_dir, "eval")
                               if args.model_dir and os.path.isdir(args.model_dir) and not args.predictions
                               and args.oracle_noise is None else "eval_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    if not args.predictions:
        with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
            for r, t in zip(rows, texts):
                f.write(json.dumps({"file_name": r["file_name"], "task": r["task"], "prediction": t}) + "\n")
    print(f"\nReport written to {os.path.join(out_dir, 'eval_report.json')}")
    return report


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate P2N predictions across all tasks.")
    p.add_argument("--model_dir", type=str, default="./p2n-model", help="Path to trained model")
    p.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"],
                   help="Image-level split to evaluate on ('all' = no filtering)")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--test_fraction", type=float, default=0.05)
    p.add_argument("--tasks", type=str, nargs="+", default=None, choices=list(TASKS))
    p.add_argument("--plot_types", type=str, nargs="+", default=None)
    p.add_argument("--num_samples", type=int, default=200, help="Samples per task (stratified by plot type)")
    p.add_argument("--max_length", type=int, default=1024, help="Max new decoder tokens")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--predictions", type=str, default=None,
                   help="Score an existing predictions.jsonl instead of running a model")
    p.add_argument("--oracle_noise", type=float, default=None,
                   help="Score perturbed ground truth instead of a model (metric sanity check)")
    p.add_argument("--oracle_drop", type=float, default=0.0,
                   help="With --oracle_noise: fraction of records to drop")
    p.add_argument("--out_dir", type=str, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.split == "all":
        args.split = None
    evaluate(args)
