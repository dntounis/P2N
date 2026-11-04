"""Structure-aware numeric metrics for P2N predictions.

The old evaluator flattened every number in the prediction and the ground
truth and zipped them by position.  One missing or extra point shifted every
subsequent pair, and predictions with *fewer* numbers were rewarded because
only ``min(len(pred), len(gt))`` values were compared.

This module instead:

1. flattens ``panels[*].data_series`` into *records* (one dict per point /
   bar / bin / parameter block),
2. groups records by their ``type`` field and optimally matches predicted to
   ground-truth records (Hungarian algorithm) using a cost that combines the
   normalised numeric error and categorical (label) agreement,
3. scores every ground-truth numeric value: an unmatched record or a missing
   field counts as a full error (1.0), so incomplete outputs are penalised,
4. reports both *normalised* errors (relative to the per-field spread of the
   ground truth, in log10 space for fields spanning >= 2 decades) and *raw*
   MAE/RMSE in data units on matched values.

Headline numbers:
  ``value_acc@{1,2,5,10}`` - share of GT numeric values recovered within that
  percentage of the field's range (recall-like, penalises misses)
  ``nmae``                - mean normalised error, capped at 1 per value
  ``record_f1``           - F1 over records matched with cost < 0.1
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from p2n_format import expand_series

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is in requirements
    linear_sum_assignment = None

TOLERANCES = (0.01, 0.02, 0.05, 0.10)
MATCH_COST_THRESHOLD = 0.10
LOG_DECADES_THRESHOLD = 2.0  # max/min >= 10**2 -> compare in log space


# ---------------------------------------------------------------------------
# Record extraction
# ---------------------------------------------------------------------------
def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def flatten_record(rec: dict, prefix: str = "") -> tuple[dict, dict]:
    """Split a record into ``(numeric_fields, categorical_fields)``.

    Nested lists/dicts are flattened with dotted keys (``triple_point.0``).
    """
    nums, cats = {}, {}
    for k, v in rec.items():
        key = f"{prefix}{k}"
        if _is_num(v):
            nums[key] = float(v)
        elif isinstance(v, (str, bool)) or v is None:
            if key != "type":
                cats[key] = v
        elif isinstance(v, dict):
            n, c = flatten_record(v, key + ".")
            nums.update(n)
            cats.update(c)
        elif isinstance(v, (list, tuple)):
            n, c = flatten_record({str(i): x for i, x in enumerate(v)}, key + ".")
            nums.update(n)
            cats.update(c)
    return nums, cats


def extract_records(obj: Any) -> list[dict]:
    """Return a list of ``{"type", "nums", "cats", "plot_type"}`` records."""
    if not isinstance(obj, dict):
        return []
    panels = obj.get("panels")
    if not isinstance(panels, list):
        panels = [obj] if "data_series" in obj else []
    out = []
    for p in panels:
        if not isinstance(p, dict):
            continue
        series = expand_series(p.get("data_series", []))
        for rec in series:
            if not isinstance(rec, dict):
                continue
            nums, cats = flatten_record(rec)
            out.append({"type": str(rec.get("type", "")), "nums": nums, "cats": cats,
                        "plot_type": p.get("plot_type")})
    return out


def axes_of(obj: Any) -> dict | None:
    if isinstance(obj, dict):
        panels = obj.get("panels")
        if isinstance(panels, list) and panels and isinstance(panels[0], dict):
            ax = panels[0].get("axes")
            return ax if isinstance(ax, dict) else None
    return None


def score_axes(pred: Any, gt: Any) -> tuple[float | None, bool | None]:
    """Mean normalised error of the 4 axis-range endpoints + scale agreement."""
    g, p = axes_of(gt), axes_of(pred)
    if not g:
        return None, None
    p = p or {}
    errs = []
    for key, scale_key in (("x_range", "x_scale"), ("y_range", "y_scale")):
        gr, pr = g.get(key), p.get(key)
        if not (isinstance(gr, list) and len(gr) == 2 and all(_is_num(v) for v in gr)):
            continue
        is_log = g.get(scale_key) == "log" and min(gr) > 0
        tg = [math.log10(v) for v in gr] if is_log else gr
        span = abs(tg[1] - tg[0]) or 1.0
        if isinstance(pr, list) and len(pr) == 2 and all(_is_num(v) for v in pr):
            if is_log and min(pr) <= 0:
                errs.extend([1.0, 1.0])
                continue
            tp = [math.log10(v) for v in pr] if is_log else pr
            errs.extend(min(abs(a - b) / span, 1.0) for a, b in zip(tp, tg))
        else:
            errs.extend([1.0, 1.0])
    scale_ok = all(_cat_eq(p.get(k), g.get(k)) for k in ("x_scale", "y_scale") if k in g)
    return (float(np.mean(errs)) if errs else None), scale_ok


def plot_type_of(obj: Any) -> str | None:
    if isinstance(obj, dict):
        panels = obj.get("panels")
        if isinstance(panels, list) and panels and isinstance(panels[0], dict):
            return panels[0].get("plot_type")
    return None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
class FieldScale:
    """Per-(type, field) transform + scale derived from the ground truth."""

    def __init__(self, values: list[float]):
        arr = np.asarray(values, dtype=float)
        self.log = bool(arr.size and np.all(arr > 0) and
                        np.log10(arr.max() / arr.min()) >= LOG_DECADES_THRESHOLD)
        t = np.log10(arr) if self.log else arr
        rng = float(t.max() - t.min()) if t.size else 0.0
        self.range = rng

    def transform(self, v: float) -> float:
        if self.log:
            return math.log10(v) if v > 0 else -math.inf
        return v

    def error(self, pred: float, gt: float) -> float:
        tg, tp = self.transform(gt), self.transform(pred)
        if not math.isfinite(tp):
            return 1.0
        scale = self.range
        if scale <= 1e-12:
            # Single value / constant field: fall back to relative error.
            scale = abs(tg) if abs(tg) > 1e-12 else 1.0
            if self.log:
                scale = 1.0  # one decade
        return min(abs(tp - tg) / scale, 1.0)


def build_scales(gt_records: list[dict]) -> dict:
    vals = defaultdict(list)
    for r in gt_records:
        for k, v in r["nums"].items():
            vals[(r["type"], k)].append(v)
    return {key: FieldScale(v) for key, v in vals.items()}


def _cat_eq(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def record_cost(gt: dict, pred: dict, scales: dict) -> float:
    errs = []
    for k, g in gt["nums"].items():
        p = pred["nums"].get(k)
        errs.append(1.0 if p is None else scales[(gt["type"], k)].error(p, g))
    num_cost = float(np.mean(errs)) if errs else 0.0
    cat_cost = 0.0
    if gt["cats"]:
        mism = sum(0 if _cat_eq(pred["cats"].get(k), v) else 1 for k, v in gt["cats"].items())
        cat_cost = mism / len(gt["cats"])
    return num_cost + cat_cost


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    # Greedy fallback
    pairs, used_r, used_c = [], set(), set()
    for idx in np.argsort(cost, axis=None):
        i, j = divmod(int(idx), cost.shape[1])
        if i not in used_r and j not in used_c:
            pairs.append((i, j))
            used_r.add(i)
            used_c.add(j)
    return pairs


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------
def score_p2n(pred: Any, gt: dict) -> dict:
    """Score one prediction against one ground-truth ``gt_parse`` dict."""
    gt_recs = extract_records(gt)
    pred_recs = extract_records(pred) if pred is not None else []
    scales = build_scales(gt_recs)

    by_type_gt, by_type_pred = defaultdict(list), defaultdict(list)
    for i, r in enumerate(gt_recs):
        by_type_gt[r["type"]].append(i)
    for j, r in enumerate(pred_recs):
        by_type_pred[r["type"]].append(j)

    matches: dict[int, int] = {}
    good_matches = 0
    for t, gidx in by_type_gt.items():
        pidx = by_type_pred.get(t, [])
        if not pidx:
            continue
        cost = np.array([[record_cost(gt_recs[i], pred_recs[j], scales) for j in pidx] for i in gidx])
        for a, b in _assign(cost):
            matches[gidx[a]] = pidx[b]
            if cost[a, b] < MATCH_COST_THRESHOLD:
                good_matches += 1

    norm_errs: list[float] = []
    raw_abs: list[float] = []
    raw_rel: list[float] = []
    per_field = defaultdict(list)
    for i, g in enumerate(gt_recs):
        p = pred_recs[matches[i]] if i in matches else None
        for k, gv in g["nums"].items():
            pv = p["nums"].get(k) if p is not None else None
            if pv is None:
                e = 1.0
            else:
                e = scales[(g["type"], k)].error(pv, gv)
                raw_abs.append(abs(pv - gv))
                if abs(gv) > 1e-12:
                    raw_rel.append(abs(pv - gv) / abs(gv))
            norm_errs.append(e)
            per_field[f"{g['type']}.{k}"].append(e)

    n_gt, n_pred = len(gt_recs), len(pred_recs)
    precision = good_matches / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = good_matches / n_gt if n_gt else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gt_pt, pred_pt = plot_type_of(gt), plot_type_of(pred)
    axes_err, axes_scale_ok = score_axes(pred, gt)
    return {
        "axes_range_err": axes_err,
        "axes_scale_correct": axes_scale_ok,
        "n_gt_records": n_gt,
        "n_pred_records": n_pred,
        "count_exact": n_gt == n_pred,
        "record_precision": precision,
        "record_recall": recall,
        "record_f1": f1,
        "plot_type_correct": (gt_pt is not None and gt_pt == pred_pt),
        "norm_errors": norm_errs,
        "raw_abs_errors": raw_abs,
        "raw_rel_errors": raw_rel,
        "per_field": dict(per_field),
    }


def _mean_not_none(vals):
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def aggregate_p2n(scores: list[dict], statuses: list[str] | None = None) -> dict:
    """Micro-average per-sample scores into a summary dict."""
    if not scores:
        return {"n": 0}
    all_err = np.array([e for s in scores for e in s["norm_errors"]], dtype=float)
    raw = np.array([e for s in scores for e in s["raw_abs_errors"]], dtype=float)
    rel = np.array([e for s in scores for e in s["raw_rel_errors"]], dtype=float)
    per_sample_mae = [float(np.mean(s["raw_abs_errors"])) for s in scores if s["raw_abs_errors"]]
    per_sample_rmse = [float(np.sqrt(np.mean(np.square(s["raw_abs_errors"])))) for s in scores
                       if s["raw_abs_errors"]]
    out = {
        "n": len(scores),
        "n_values": int(all_err.size),
        "nmae": float(all_err.mean()) if all_err.size else None,
        **{f"value_acc@{int(t*100)}": float((all_err <= t).mean()) if all_err.size else None
           for t in TOLERANCES},
        "record_precision": float(np.mean([s["record_precision"] for s in scores])),
        "record_recall": float(np.mean([s["record_recall"] for s in scores])),
        "record_f1": float(np.mean([s["record_f1"] for s in scores])),
        "count_exact_acc": float(np.mean([s["count_exact"] for s in scores])),
        "plot_type_acc": float(np.mean([s["plot_type_correct"] for s in scores])),
        "axes_range_err": _mean_not_none(s.get("axes_range_err") for s in scores),
        "axes_scale_acc": _mean_not_none(s.get("axes_scale_correct") for s in scores),
        # Raw-unit errors on matched values (mixes units across plot types, so
        # the per-sample median is the more meaningful summary).
        "matched_mae": float(raw.mean()) if raw.size else None,
        "matched_rmse": float(np.sqrt(np.mean(raw ** 2))) if raw.size else None,
        "median_sample_mae": float(np.median(per_sample_mae)) if per_sample_mae else None,
        "median_sample_rmse": float(np.median(per_sample_rmse)) if per_sample_rmse else None,
        "median_rel_error": float(np.median(rel)) if rel.size else None,
    }
    if statuses is not None:
        n = len(statuses)
        out["json_valid_rate"] = sum(s == "valid" for s in statuses) / n
        out["json_parse_rate"] = sum(s in ("valid", "repaired") for s in statuses) / n
    return out


# ---------------------------------------------------------------------------
# Auxiliary tasks
# ---------------------------------------------------------------------------
AXIS_FIELDS = ("x_scale", "y_scale", "x_label", "y_label", "x_inverted", "y_inverted")
ELEMENT_BOOL_FIELDS = ("has_legend", "has_colorbar", "has_error_bars", "has_grid")


def score_fields(pred: Any, gt: dict, fields) -> dict:
    pred = pred if isinstance(pred, dict) else {}
    return {f: _cat_eq(pred.get(f), gt.get(f)) for f in fields if f in gt}


def score_axis_info(pred: Any, gt: dict) -> dict:
    s = score_fields(pred, gt, AXIS_FIELDS)
    s["all_correct"] = all(s.values()) if s else False
    return s


def score_element_count(pred: Any, gt: dict) -> dict:
    s = score_fields(pred, gt, ELEMENT_BOOL_FIELDS + ("plot_type",))
    pred = pred if isinstance(pred, dict) else {}
    if _is_num(pred.get("n_series")) and _is_num(gt.get("n_series")):
        s["n_series_abs_err"] = abs(pred["n_series"] - gt["n_series"])
    else:
        s["n_series_abs_err"] = None
    return s


def aggregate_fields(scores: list[dict]) -> dict:
    if not scores:
        return {"n": 0}
    keys = sorted({k for s in scores for k in s})
    out = {"n": len(scores)}
    for k in keys:
        vals = [s[k] for s in scores if k in s and s[k] is not None]
        if not vals:
            continue
        if k.endswith("abs_err"):
            out[k.replace("abs_err", "mae")] = float(np.mean(vals))
        else:
            out[f"{k}_acc"] = float(np.mean([bool(v) for v in vals]))
    return out
