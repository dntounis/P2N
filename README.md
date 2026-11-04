# P2N: Plot to Numbers

P2N (Plot 2 Numbers) is an end-to-end framework for parsing scientific plots directly into structured numerical data (JSON). It fine-tunes the Donut vision-encoder-decoder (`naver-clova-ix/donut-base`) for OCR-free chart derendering.

```
plot image ──► DonutSwin encoder ──► MBart decoder ──► <s_p2n>{"panels":[{"plot_type":..,"axes":{..},"data_series":[..]}]}</s_p2n>
```

## Pipeline overview

| Stage | Script | Notes |
|---|---|---|
| Data generation | `generate_data.py` | 131 Matplotlib plot generators, HF `imagefolder` layout (`images/<plot_type>/image_XXXXX.jpg` + `metadata.jsonl`), optional degradation + auxiliary tasks, reproducible with `--seed` |
| Data audit | `tools/audit_generators.py`, `tools/target_stats.py` | Per-generator failure rate, target size, *constant-target detection*, token-length distribution |
| Training | `train.py` | Seq2SeqTrainer, DDP via `torchrun`, compact targets, dynamic padding, generation-based validation metrics |
| Evaluation | `evaluate.py` (+ `p2n_metrics.py`) | Structure-aware numeric metrics on a held-out test split |
| Inference | `inference.py`, `backend_app.py`, `frontend/` | Batched generation, JSON repair, compact → verbose expansion |

Shared logic lives in `p2n_format.py` (task tokens, target serialisation, JSON repair, image-level splits) and `dataset.py` (dataset + collator).

## Setup

```bash
pip install -r requirements.txt
```

## 1. Generate synthetic data

```bash
# 50k images + auxiliary axis/element tasks, 30% degraded, reproducible
python generate_data.py --samples 50000 --output_dir data --aux_tasks --degrade_fraction 0.3 --seed 1

# Restrict / exclude plot types (e.g. a curriculum stage on point-level plots)
python generate_data.py --samples 20000 --output_dir data_core --plot_types scatter line_plot bar histogram --seed 2
python generate_data.py --list_plot_types
```

Each panel's ground truth now also records the visible axis ranges/scales (`"axes": {"x_range", "y_range", "x_scale", "y_scale"}`), which the model learns to read and the evaluator scores.

Audit the generators (failures, target size, near-constant targets that can only be memorised):

```bash
python tools/audit_generators.py --runs 4 --out audit.json
python tools/target_stats.py --dataset_path data --max_length 1024
```

## 2. Train

Targets are serialised in a lossless **compact columnar format**: consecutive records sharing a type are grouped, constant fields are hoisted and the remaining fields become a table:

```json
{"type":"scatter","x_log":false,"y_log":false,"cols":["x","y"],"rows":[[12.3,45.6],[78.9,1.2]]}
```

This is typically 2–3× fewer decoder tokens than one dict per point, so far fewer targets exceed `--max_length` (over-length targets are dropped, not truncated — a truncated target never contains the end token and teaches the model not to stop).

Splits are deterministic and **image-level** (hash of the file path): 90% train / 5% val / 5% test, so auxiliary-task rows never leak an evaluation image into training.

```bash
# Multi-GPU (e.g. 4x A100)
torchrun --nproc_per_node=4 train.py \
    --dataset_path data \
    --output_dir ./p2n-model \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --image_size 1024 1024 \
    --max_length 1024 \
    --num_workers 8 \
    --gen_eval_samples 64
```

Useful flags:

* `--image_size H W` – encoder resolution (default 1024×1024; donut-base's native 2560×1920 is ~4.7× more expensive and unnecessary for plots).
* `--gen_eval_samples N` – at every evaluation, decode N validation images and log `value_acc@5`, `nmae`, JSON parse rate (also written to `<output_dir>/gen_eval_log.jsonl`).
* `--tasks p2n` / `--plot_types ...` – single-task or curriculum training.
* `--target_format verbose` – the old one-dict-per-record format.
* Mixed precision is picked automatically (bf16 if supported, else fp16; `--fp32` to disable).

## 3. Evaluate

```bash
# Held-out test split, 200 samples per task, stratified over plot types
python evaluate.py --model_dir ./p2n-model --dataset_path data --num_samples 200

# Re-score saved predictions / sanity-check the metric with noisy ground truth
python evaluate.py --predictions ./p2n-model/eval/predictions.jsonl --dataset_path data
python evaluate.py --oracle_noise 0.02 --dataset_path data
```

Metrics for the main task (`p2n_metrics.py`):

| Metric | Meaning |
|---|---|
| `json_parse_rate` | outputs that parse as JSON (after repairing truncated output) |
| `value_acc@{1,2,5,10}` | share of **ground-truth** numeric values recovered within 1/2/5/10% of that field's range — missing values count as wrong |
| `nmae` | mean normalised absolute error (capped at 1 per value; log10 space for fields spanning ≥ 2 decades) |
| `record_precision/recall/f1` | points/bars/bins matched with cost < 0.1 |
| `count_exact_acc` | predicted #records == ground truth |
| `axes_range_err`, `axes_scale_acc` | axis-range reading error (fraction of axis span), lin/log scale accuracy |
| `median_sample_mae/rmse` | raw MAE/RMSE in data units on matched values |

Records are matched per `type` with the Hungarian algorithm (cost = normalised numeric error + label mismatch), so point order doesn't matter and dropping or hallucinating points is penalised. Auxiliary tasks report per-field accuracy.

## 4. Inference

```bash
python inference.py --model_dir ./p2n-model --image plot.png --task all
python inference.py --model_dir ./p2n-model --image_dir ./test_images/ --batch_size 8 --output preds.json
```

Outputs are parsed (with truncated-JSON repair) and compact series are expanded back to one dict per record.

## Offline smoke test (no GPU / no Hugging Face Hub access)

`tools/make_tiny_model.py` builds a tiny randomly-initialised model with the same architecture classes as donut-base (DonutSwin + MBart + XLM-R-style tokenizer) to exercise the whole pipeline on CPU:

```bash
python generate_data.py --samples 300 --output_dir data_smoke --plot_types bar --seed 0 --degrade_fraction 0
python tools/make_tiny_model.py --dataset_path data_smoke --out tiny-donut --image_size 320 320
python train.py --model_id tiny-donut --dataset_path data_smoke --output_dir smoke-model \
    --image_size 320 320 --max_steps 200 --batch_size 8 --gradient_accumulation_steps 1 \
    --learning_rate 1e-3 --eval_steps 100 --save_steps 1000 --num_workers 2 --gen_eval_samples 16
python evaluate.py --model_dir smoke-model --dataset_path data_smoke --tasks p2n --num_samples 30
```

## Tests

```bash
python -m pytest -q tests
```
