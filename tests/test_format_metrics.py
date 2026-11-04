import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2n_format import (  # noqa: E402
    build_target_text,
    canonical_target,
    compact_series,
    expand_p2n,
    expand_series,
    parse_ground_truth,
    parse_prediction,
    repair_json,
    round_sig,
    split_of,
)
from p2n_metrics import aggregate_p2n, extract_records, score_axes, score_p2n  # noqa: E402


def scatter_gt(n=10, seed=0):
    rng = random.Random(seed)
    pts = [{"type": "scatter", "x": round(rng.uniform(0, 100), 1), "y": round(rng.uniform(0, 100), 1),
            "x_log": False, "y_log": False} for _ in range(n)]
    return {"figure_type": "single_panel", "image_quality": "clean", "degradation_effects": [],
            "panels": [{"panel_id": "A", "plot_type": "scatter",
                        "axes": {"x_range": [0.0, 100.0], "y_range": [0.0, 100.0],
                                 "x_scale": "linear", "y_scale": "linear"},
                        "data_series": pts}]}


# ---------------------------------------------------------------- format
def test_round_sig():
    assert round_sig(123456.0, 4) == 123500.0
    assert round_sig(0.000123456, 3) == 0.000123
    assert round_sig(-2.5e34, 2) == -2.5e34
    assert round_sig(0.0) == 0.0
    assert round_sig(7) == 7


def test_compact_roundtrip():
    gt = scatter_gt()
    series = gt["panels"][0]["data_series"]
    comp = compact_series(series)
    assert len(comp) == 1 and comp[0]["cols"] == ["x", "y"]
    assert comp[0]["x_log"] is False
    assert expand_series(comp) == series


def test_compact_mixed_types_roundtrip():
    series = [{"type": "fit_data", "x": 1.0, "y": 2.0}, {"type": "fit_data", "x": 2.0, "y": 3.0},
              {"type": "fit_params", "a": 1.0, "b": 0.5}]
    comp = compact_series(series)
    assert comp[-1] == {"type": "fit_params", "a": 1.0, "b": 0.5}
    assert expand_series(comp) == series


def test_compact_is_shorter():
    gt = scatter_gt(20)
    _, compact = build_target_text({"gt_parse": gt, "task": "p2n"}, target_format="compact")
    _, verbose = build_target_text({"gt_parse": gt, "task": "p2n"}, target_format="verbose")
    assert len(compact) < 0.6 * len(verbose)
    assert "degradation_effects" not in compact and "image_quality" not in compact
    assert expand_p2n(json.loads(compact)) == json.loads(verbose)


def test_parse_ground_truth_wrapped_and_bare():
    assert parse_ground_truth(json.dumps({"gt_parse": {"a": 1}, "task": "axis_info"})) == ("axis_info", {"a": 1})
    assert parse_ground_truth(json.dumps({"a": 1})) == ("p2n", {"a": 1})
    assert parse_ground_truth("not json") == ("p2n", {})


def test_aux_target_drops_task_type():
    t = canonical_target("axis_info", {"task_type": "axis_info", "x_scale": "log"})
    assert t == {"x_scale": "log"}


def test_repair_truncated_json():
    full = json.dumps({"panels": [{"plot_type": "bar", "data_series": [{"v": 1}, {"v": 2}, {"v": 3}]}]})
    truncated = full[: full.index('{"v": 3') + 4]
    obj, status = parse_prediction(truncated)
    assert status == "repaired"
    assert obj["panels"][0]["data_series"] == [{"v": 1}, {"v": 2}]
    assert parse_prediction(full)[1] == "valid"
    assert parse_prediction("garbage")[1] == "invalid"
    assert repair_json('{"a": "unterminated') is None or json.loads(repair_json('{"a": "unterminated'))


def test_parse_strips_task_tokens():
    obj, status = parse_prediction('<s_p2n>{"a":1}</s_p2n>')
    assert obj == {"a": 1} and status == "valid"


def test_split_is_image_level_and_deterministic():
    names = [f"images/scatter/image_{i:05d}.jpg" for i in range(5000)]
    splits = [split_of(n) for n in names]
    assert splits == [split_of(n) for n in names]
    frac_test = splits.count("test") / len(splits)
    frac_val = splits.count("val") / len(splits)
    assert 0.03 < frac_test < 0.07 and 0.03 < frac_val < 0.07


# ---------------------------------------------------------------- metrics
def test_perfect_prediction_scores_perfectly():
    gt = canonical_target("p2n", scatter_gt(), target_format="verbose")
    s = score_p2n(gt, gt)
    agg = aggregate_p2n([s], ["valid"])
    assert agg["value_acc@1"] == 1.0 and agg["nmae"] == 0.0 and agg["record_f1"] == 1.0
    assert agg["axes_range_err"] == 0.0 and agg["count_exact_acc"] == 1.0


def test_permutation_invariance_and_compact_input():
    gt = canonical_target("p2n", scatter_gt(), target_format="verbose")
    pred = json.loads(json.dumps(gt))
    random.Random(1).shuffle(pred["panels"][0]["data_series"])
    pred = canonical_target("p2n", pred, target_format="compact")  # model emits compact form
    s = score_p2n(pred, gt)
    assert s["record_f1"] == 1.0 and max(s["norm_errors"]) == 0.0


def test_missing_points_are_penalised():
    gt = canonical_target("p2n", scatter_gt(10), target_format="verbose")
    pred = json.loads(json.dumps(gt))
    pred["panels"][0]["data_series"] = pred["panels"][0]["data_series"][:5]
    s = score_p2n(pred, gt)
    assert s["record_recall"] == 0.5 and s["record_precision"] == 1.0
    agg = aggregate_p2n([s])
    assert agg["value_acc@5"] == pytest.approx(0.5)


def test_noise_degrades_monotonically():
    gt = canonical_target("p2n", scatter_gt(30), target_format="verbose")
    accs = []
    for noise in (0.0, 0.02, 0.2):
        rng = random.Random(0)
        pred = json.loads(json.dumps(gt))
        for r in pred["panels"][0]["data_series"]:
            r["x"] += rng.gauss(0, noise * 100)
            r["y"] += rng.gauss(0, noise * 100)
        accs.append(aggregate_p2n([score_p2n(pred, gt)])["value_acc@5"])
    assert accs[0] == 1.0 and accs[0] >= accs[1] > accs[2]


def test_log_fields_use_log_space():
    gt = {"panels": [{"plot_type": "x", "data_series": [
        {"type": "p", "v": 0.01}, {"type": "p", "v": 1.0}, {"type": "p", "v": 1000.0}]}]}
    pred = {"panels": [{"plot_type": "x", "data_series": [
        {"type": "p", "v": 0.011}, {"type": "p", "v": 1.1}, {"type": "p", "v": 1100.0}]}]}
    s = score_p2n(pred, gt)
    # 10% relative error = 0.041 decades over a 5-decade range -> < 1% normalised
    assert max(s["norm_errors"]) < 0.01


def test_invalid_prediction_scores_zero():
    gt = canonical_target("p2n", scatter_gt(), target_format="verbose")
    s = score_p2n(None, gt)
    assert s["record_recall"] == 0.0 and min(s["norm_errors"]) == 1.0


def test_labels_drive_matching():
    gt = {"panels": [{"plot_type": "bar", "data_series": [
        {"type": "bar", "category": "a", "value": 10.0}, {"type": "bar", "category": "b", "value": 50.0}]}]}
    pred = {"panels": [{"plot_type": "bar", "data_series": [
        {"type": "bar", "category": "b", "value": 50.0}, {"type": "bar", "category": "a", "value": 10.0}]}]}
    assert score_p2n(pred, gt)["record_f1"] == 1.0


def test_axes_scoring_log():
    gt = {"panels": [{"axes": {"x_range": [1.0, 1000.0], "y_range": [0.0, 1.0],
                               "x_scale": "log", "y_scale": "linear"}}]}
    pred = {"panels": [{"axes": {"x_range": [1.0, 1000.0], "y_range": [0.0, 1.0],
                                 "x_scale": "log", "y_scale": "linear"}}]}
    err, ok = score_axes(pred, gt)
    assert err == 0.0 and ok


def test_extract_records_flattens_nested():
    recs = extract_records({"panels": [{"data_series": [{"type": "pd", "tp": [273.16, 0.006]}]}]})
    assert recs[0]["nums"] == {"tp.0": 273.16, "tp.1": 0.006}
