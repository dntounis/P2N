# P2N status and next steps (2026-11-04)

## What changed in this pass

### Critical bugs fixed
1. **Double label shift in the loss.** With transformers ≥ 4.46, `VisionEncoderDecoderModel` computes its loss with `ForCausalLMLoss`, which shifts labels by one more position. The model was trained to predict the token *after next*: training loss went down, but greedy generation was garbage. This affected the original code too on any transformers newer than the pinned 4.44.2. `train.py` now computes token cross-entropy itself (`P2NTrainer`), and a regression test covers it.
2. **Auxiliary-task prompt mismatch.** All tasks were trained with `<s_p2n>` as the first decoder token, but inference prompted with `<s_axis_info>` / `<s_element_count>`. Sequences are now `[<s_task>] body [</s_task>, </s>]` with explicit `decoder_input_ids`, identical at training and inference.
3. **Train/eval leakage.** The aux-task rows of an image could land in the train split while its main row was evaluated. Splits are now image-level and deterministic (`p2n_format.split_of`), with 90/5/5 train/val/test.
4. **Truncated targets.** Long targets were cut at `max_length`, so they never contained the end token. They are now dropped, and the dataset reports how many and which plot types.
5. **Positional number matching in evaluation.** The old evaluator zipped numbers by position (one missing point shifted all later ones) and rewarded shorter outputs. Replaced by Hungarian matching per record type; see `p2n_metrics.py`.

### Data quality
- `tools/audit_generators.py` showed only ~20/131 generators emitted point-level ground truth. ~95 emitted a single summary record, and 42 had constant targets (e.g. `bump_hunt` always peak 125), which can only be memorised.
- Ground truth is now augmented with **the data actually drawn on the axes** (curves subsampled at 12 points, marker series, small scatters, bars; only points inside the visible limits; errorbar caps excluded). This applies to generators that return fewer than 3 records. Result: **99/131 plot types now have point-level GT**, with mean numeric values per target up from ~11 to ~35. The cap is 64 records per plot.
- Every panel records its visible **axis ranges and scales**, which are scored separately (`axes_range_err`, `axes_scale_acc`).
- Compact columnar target format, lossless, about 2–3× fewer tokens.
- `--seed` gives byte-identical datasets, and `--plot_types` / `--exclude_plot_types` enable curricula.

### Verified offline (CPU, tiny random Donut-architecture model, 4k bar charts, 2400 steps)

| step | parse rate | value_acc@5 | nMAE | axes range err |
|---|---|---|---|---|
| 400 | 0.00 | 0.000 | 1.00 | 1.00 |
| 800 | 1.00 | 0.043 | 0.68 | 0.125 |
| 1200 | 1.00 | 0.072 | 0.43 | 0.131 |
| 2400 | 1.00 | 0.047 | 0.58 | 0.124 |

A 7M-parameter model trained from scratch on CPU cannot learn to read values well. This run validates the pipeline (format → training → generation → parsing → scoring), not the model's capability. The real model is donut-base on GPUs.

## Recommended next runs (on the cluster)

1. **Sanity check, ~1 GPU-hour.** `python tools/target_stats.py --dataset_path data` with the real donut-base tokenizer. Check the `unk_tokens` count (should be 0) and the over-length share at `--max_length 1024`.
2. **Stage 1 curriculum, point-level types.** Train on scatter, line_plot, bar, grouped_bar, histogram, fit, errorbar-like types (`--plot_types ...`), 20k images, `--tasks p2n`, 1024², about 5 epochs. Watch `gen_value_acc@5` in `gen_eval_log.jsonl`.
3. **Stage 2, full mix.** 50k+ images, all tasks, resume from stage 1 (`--model_id <stage1 dir>`).
4. **Evaluate.** `python evaluate.py --model_dir ... --num_samples 300`. Use the per-plot-type table to decide which generators need targeted fixes or more weight.

`scripts/run_pipeline.sh` wraps generate → audit → train → evaluate, and is SLURM-ready.

## Known gaps / ideas
- 32 plot types (heatmaps, contours, spectrograms, hexbin, sky maps, vector fields) still have summary-only GT. Readable targets for these would be colorbar range plus a coarse grid of sampled values.
- Dense scatters (> 60 points) are not listed point by point. A density or extent summary may be the right target.
- There is no real-world test set yet. A few dozen digitised real plots in the same `metadata.jsonl` format would be the most valuable eval addition; `evaluate.py` works on any such folder with `--split all`.
- Number tokenization in the donut-base sentencepiece vocab is irregular. If numeric accuracy plateaus, try fixed-precision formatting or digit-level tokens.
