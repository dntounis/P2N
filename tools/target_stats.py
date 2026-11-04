"""Decoder-target length statistics (choose --max_length / spot verbose generators).

Tokenises every training target with the model's tokenizer and reports, per
plot type, the median / p95 / max token lengths and the share of targets
that would be dropped at a given --max_length, for both target formats.

Usage:
    python tools/target_stats.py --dataset_path data --tokenizer naver-clova-ix/donut-base --max_length 1024
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from p2n_format import build_target_text, load_metadata  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="data")
    ap.add_argument("--tokenizer", default="naver-clova-ix/donut-base")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--tasks", nargs="+", default=["p2n"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_metadata(args.dataset_path, tasks=args.tasks)[: args.limit]
    stats = {}
    for fmt in ("compact", "verbose"):
        bodies = [build_target_text(r["ground_truth"], target_format=fmt)[1] for r in rows]
        lens = [len(x) + 3 for x in tok(bodies, add_special_tokens=False).input_ids]
        unk = sum(ids.count(tok.unk_token_id) for ids in tok(bodies[:2000], add_special_tokens=False).input_ids)
        by_pt = defaultdict(list)
        for r, n in zip(rows, lens):
            by_pt[r["file_name"].split("/")[1]].append(n)
        stats[fmt] = (lens, by_pt, unk)

    print(f"{len(rows)} targets, max_length={args.max_length}")
    for fmt, (lens, _, unk) in stats.items():
        a = np.array(lens)
        print(f"  {fmt:8s} p50={np.median(a):.0f} p90={np.percentile(a, 90):.0f} "
              f"p99={np.percentile(a, 99):.0f} max={a.max()} over={np.mean(a > args.max_length) * 100:.1f}% "
              f"unk_tokens(first 2k)={unk}")
    print(f"\n{'plot_type':28s} {'n':>5s} {'p50':>6s} {'p95':>6s} {'max':>6s} {'over%':>6s}  (compact)")
    _, by_pt, _ = stats["compact"]
    for pt, v in sorted(by_pt.items(), key=lambda kv: -np.median(kv[1])):
        a = np.array(v)
        print(f"{pt:28s} {len(a):5d} {np.median(a):6.0f} {np.percentile(a, 95):6.0f} {a.max():6d} "
              f"{np.mean(a > args.max_length) * 100:6.1f}")


if __name__ == "__main__":
    main()
