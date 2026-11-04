"""Integration tests for data generation and the dataset/collator (CPU, offline)."""
import json
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import generate_data as gd  # noqa: E402
from p2n_format import build_target_text, expand_p2n, load_metadata, parse_ground_truth  # noqa: E402


@pytest.mark.parametrize("plot_type", gd.PLOT_TYPES)
def test_generator_renders_and_serialises(plot_type):
    random.seed(3)
    np.random.seed(3)
    fig, data_points, meta, axes = gd.render_sample(plot_type, figsize=(5, 5))
    plt.close(fig)
    assert isinstance(data_points, list) and data_points
    json.dumps(data_points)  # must be JSON serialisable (no numpy types)
    assert meta["plot_type"] == plot_type


def test_auto_series_matches_drawn_data():
    fig, ax = plt.subplots()
    xs = np.arange(5, dtype=float)
    ax.plot(xs, xs ** 2, "o", label="sq")
    ax.bar([10, 11], [3.0, 4.0], label="b")
    recs = gd.extract_drawn_series(ax)
    plt.close(fig)
    markers = [(r["x"], r["y"]) for r in recs if r["type"] == "marker"]
    bars = [(r["pos"], r["value"]) for r in recs if r["type"] == "bar_top"]
    assert markers == [(x, x ** 2) for x in xs]
    assert bars == [(10.0, 3.0), (11.0, 4.0)]
    assert all(r["series"] in ("sq", "b") for r in recs)


def test_auto_series_skips_errorbar_caps():
    fig, ax = plt.subplots()
    ax.errorbar([1, 2, 3], [1.0, 2.0, 3.0], yerr=0.5, fmt="ko-", capsize=3)
    recs = gd.extract_drawn_series(ax)
    plt.close(fig)
    assert [r["y"] for r in recs] == [1.0, 2.0, 3.0]


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    gd.generate_plot(str(out), 12, degrade_fraction=0.5, aux_tasks=True, num_cpus=2,
                     seed=5, plot_types=["scatter", "bar", "line_plot"])
    return str(out)


def test_generation_layout_and_reproducibility(small_dataset, tmp_path):
    rows = load_metadata(small_dataset)
    assert len(rows) == 36  # 12 images x (p2n + 2 aux tasks)
    for r in rows:
        parts = r["file_name"].split("/")
        assert parts[0] == "images" and parts[1] in ("scatter", "bar", "line_plot")
        assert os.path.exists(r["image_path"])
    gd.generate_plot(str(tmp_path), 12, degrade_fraction=0.5, aux_tasks=True, num_cpus=2,
                     seed=5, plot_types=["scatter", "bar", "line_plot"])
    a = sorted(open(os.path.join(small_dataset, "metadata.jsonl")).read().splitlines())
    b = sorted(open(os.path.join(tmp_path, "metadata.jsonl")).read().splitlines())
    assert a == b


def test_p2n_ground_truth_has_axes(small_dataset):
    for r in load_metadata(small_dataset, tasks=["p2n"]):
        _, gt = parse_ground_truth(r["ground_truth"])
        axes = gt["panels"][0]["axes"]
        assert len(axes["x_range"]) == 2 and axes["x_scale"] in ("linear", "log")
        _, body = build_target_text(r["ground_truth"])
        assert expand_p2n(json.loads(body))["panels"][0]["data_series"] == gt["panels"][0]["data_series"]


def test_dataset_and_collator(small_dataset):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import DonutImageProcessor, DonutProcessor

    from dataset import P2NCollator, P2NDataset, add_task_tokens, task_token_ids
    from tools.make_tiny_model import build_tokenizer

    tok = build_tokenizer(small_dataset, 300)
    proc = DonutProcessor(image_processor=DonutImageProcessor(size={"height": 64, "width": 64}),
                          tokenizer=tok)
    add_task_tokens(proc)
    ds = P2NDataset(small_dataset, proc, split=None, max_length=4096, verbose=False)
    assert len(ds) == 36
    batch = P2NCollator(tok.pad_token_id)([ds[0], ds[1], ds[2]])
    assert batch["pixel_values"].shape[1:] == (3, 64, 64)
    dec, lab = batch["decoder_input_ids"], batch["labels"]
    assert dec.shape == lab.shape
    for i in range(3):
        task = ds.rows[i]["task"]
        start_id, end_id = task_token_ids(tok, task)
        n = int(batch["decoder_attention_mask"][i].sum())
        assert dec[i, 0].item() == start_id  # prompt = task token, as at inference
        assert lab[i, n - 1].item() == tok.eos_token_id
        assert lab[i, n - 2].item() == end_id
        assert (dec[i, 1:n] == lab[i, : n - 1]).all()  # teacher-forcing shift
        assert (lab[i, n:] == -100).all()

    # Over-length targets are dropped, not truncated.
    short = P2NDataset(small_dataset, proc, split=None, max_length=8, verbose=False)
    assert len(short) < len(ds)


def test_loss_has_no_extra_label_shift():
    """Regression: transformers' ForCausalLMLoss shifts labels again; our loss must not."""
    torch = pytest.importorskip("torch")
    from train import token_cross_entropy

    labels = torch.tensor([[3, 1, 4, 1, -100]])
    logits = torch.full((1, 5, 6), -20.0)
    for t, y in enumerate(labels[0].tolist()):
        if y >= 0:
            logits[0, t, y] = 20.0  # logits[t] predicts labels[t]
    assert token_cross_entropy(logits, labels).item() < 1e-6


def test_image_grid_extraction():
    fig, ax = plt.subplots()
    data = np.arange(16, dtype=float).reshape(4, 4)
    ax.imshow(data, vmin=0, vmax=15)
    recs = gd.extract_drawn_series(ax)
    plt.close(fig)
    assert recs[0] == {"type": "color_range", "vmin": 0.0, "vmax": 15.0}
    grid = {(r["row"], r["col"]): r["value"] for r in recs[1:]}
    assert grid[(0, 0)] == 0.0 and grid[(3, 3)] == 15.0 and len(grid) == 16
