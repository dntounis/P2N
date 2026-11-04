"""Audit every plot generator in generate_data.py.

For each plot type, renders ``--runs`` samples (with different seeds) and
reports:

  * ``fail``      - generator exceptions
  * ``recs``      - mean number of data_series records
  * ``nums``      - mean number of numeric values in the target
  * ``chars``     - mean length of the compact JSON training target
  * ``tokens``    - mean token length (only with --tokenizer)
  * ``varying``   - share of numeric fields whose value changes between runs.
                    0.0 means the target is constant, i.e. the model can only
                    memorise it rather than read it from the image.

Usage:
    python tools/audit_generators.py --runs 4 --out audit.json [--tokenizer path]
"""
import argparse
import json
import os
import sys
import random
import multiprocessing
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import generate_data as gd  # noqa: E402
from p2n_format import canonical_target, to_json  # noqa: E402
from p2n_metrics import extract_records  # noqa: E402


def _audit_one(args):
    plot_type, runs, seed = args
    out = {"plot_type": plot_type, "fail": 0, "errors": [], "targets": []}
    for r in range(runs):
        random.seed(seed + r)
        np.random.seed(seed + r)
        try:
            fig, data_points, _, axes_info = gd.render_sample(plot_type, figsize=(6, 6))
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            out["fail"] += 1
            out["errors"].append(repr(e)[:200])
            continue
        panel = {"panel_id": "A", "plot_type": plot_type}
        if axes_info:
            panel["axes"] = axes_info
        panel["data_series"] = data_points
        gt = {"figure_type": "single_panel", "panels": [panel]}
        out["targets"].append(to_json(canonical_target("p2n", gt)))
    return out


def summarize(res, tokenizer=None):
    targets = [json.loads(t) for t in res["targets"]]
    recs = [extract_records(t) for t in targets]
    n_recs = [len(r) for r in recs]
    n_nums = [sum(len(x["nums"]) for x in r) for r in recs]
    chars = [len(t) for t in res["targets"]]
    # Field variability across runs: compare values of (type, field, idx)
    values = defaultdict(set)
    for r in recs:
        counter = defaultdict(int)
        for x in r:
            idx = counter[x["type"]]
            counter[x["type"]] += 1
            for k, v in x["nums"].items():
                values[(x["type"], k, idx)].add(round(v, 6))
    varying = (np.mean([len(v) > 1 for v in values.values()]) if values and len(recs) > 1 else None)
    s = {
        "plot_type": res["plot_type"],
        "fail": res["fail"],
        "recs": float(np.mean(n_recs)) if n_recs else 0,
        "nums": float(np.mean(n_nums)) if n_nums else 0,
        "chars": float(np.mean(chars)) if chars else 0,
        "max_chars": max(chars) if chars else 0,
        "varying": None if varying is None else float(varying),
        "errors": res["errors"][:2],
    }
    if tokenizer is not None and res["targets"]:
        toks = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in res["targets"]]
        s["tokens"] = float(np.mean(toks))
        s["max_tokens"] = int(max(toks))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpus", type=int, default=None)
    ap.add_argument("--plot_types", nargs="+", default=None)
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="Optional path/name of a tokenizer to measure target token lengths.")
    ap.add_argument("--out", type=str, default=None, help="Write the full audit as JSON here.")
    args = ap.parse_args()

    types = gd.resolve_plot_types(args.plot_types)
    jobs = [(t, args.runs, args.seed) for t in types]
    with multiprocessing.Pool(args.cpus or len(os.sched_getaffinity(0))) as pool:
        raw = pool.map(_audit_one, jobs, chunksize=1)

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [summarize(r, tok) for r in raw]

    hdr = f"{'plot_type':28s} {'fail':>4s} {'recs':>6s} {'nums':>7s} {'chars':>7s} {'max':>6s} {'vary':>5s}"
    if tok:
        hdr += f" {'tok':>6s} {'maxtok':>6s}"
    print(hdr)
    for s in sorted(rows, key=lambda s: -s["chars"]):
        vary = "  -  " if s["varying"] is None else f"{s['varying']:.2f}"
        line = (f"{s['plot_type']:28s} {s['fail']:4d} {s['recs']:6.1f} {s['nums']:7.1f} "
                f"{s['chars']:7.0f} {s['max_chars']:6d} {vary:>5s}")
        if tok and "tokens" in s:
            line += f" {s['tokens']:6.0f} {s['max_tokens']:6d}"
        print(line)

    failing = [s["plot_type"] for s in rows if s["fail"]]
    constant = [s["plot_type"] for s in rows if s["varying"] is not None and s["varying"] < 0.2]
    print(f"\n{len(rows)} plot types | failing: {failing}")
    print(f"near-constant targets (<20% of numeric fields vary): {constant}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
