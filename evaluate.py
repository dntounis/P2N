"""P2N Evaluation — Multi-task evaluation with MAE/RMSE metrics.

Evaluates the trained P2N model on three tasks:
  1. p2n (main): Full data extraction — measures JSON parse rate and numerical MAE
  2. axis_info: Axis detection — measures scale/label accuracy
  3. element_count: Element detection — measures boolean field accuracy

Usage:
    python evaluate.py --model_dir ./p2n-model --dataset_path data --num_samples 100
"""
import json
import os
import re
import glob
import random
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from dataset import TASK_TOKENS


def load_model(model_dir):
    """Load trained model and processor."""
    processor = DonutProcessor.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use accelerate for automatic multi-GPU inference if available
    try:
        import accelerate
        model = VisionEncoderDecoderModel.from_pretrained(model_dir, device_map="auto")
    except ImportError:
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        model.to(device)
        
    model.eval()
    return model, processor, device


def run_inference(model, processor, device, image_path, task="p2n", max_length=1536):
    """Run inference on a single image for a given task."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    start_tok, end_tok = TASK_TOKENS.get(task, TASK_TOKENS["p2n"])
    decoder_input_ids = processor.tokenizer(
        start_tok, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    
    input_device = next(model.parameters()).device if hasattr(model, 'hf_device_map') else device
    pixel_values = pixel_values.to(input_device)
    decoder_input_ids = decoder_input_ids.to(input_device)
    
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    
    sequence = processor.batch_decode(outputs.sequences)[0]
    # Clean up special tokens
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")
    sequence = sequence.replace(start_tok, "").replace(end_tok, "").strip()
    
    return sequence


def extract_numbers(obj):
    """Recursively extract all numeric values from a nested dict/list."""
    numbers = []
    if isinstance(obj, (int, float)):
        numbers.append(float(obj))
    elif isinstance(obj, list):
        for item in obj:
            numbers.extend(extract_numbers(item))
    elif isinstance(obj, dict):
        for val in obj.values():
            numbers.extend(extract_numbers(val))
    return numbers


def compute_numerical_metrics(pred_json, gt_json):
    """Compute MAE and RMSE between predicted and ground truth numerical values."""
    pred_nums = extract_numbers(pred_json)
    gt_nums = extract_numbers(gt_json)
    
    if not gt_nums or not pred_nums:
        return None, None
    
    # Match by position (assumes same structure)
    n = min(len(pred_nums), len(gt_nums))
    if n == 0:
        return None, None
    
    pred_arr = np.array(pred_nums[:n])
    gt_arr = np.array(gt_nums[:n])
    
    mae = float(np.mean(np.abs(pred_arr - gt_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - gt_arr) ** 2)))
    
    return mae, rmse


def evaluate_p2n(model, processor, device, samples, max_length):
    """Evaluate full plot-to-numbers extraction."""
    results = {
        "total": 0, "json_valid": 0,
        "mae_values": [], "rmse_values": [],
        "per_type": {},
    }
    
    for sample in samples:
        gt_text = sample["ground_truth"]
        image_path = sample["image_path"]
        
        try:
            gt_dict = json.loads(gt_text)
            gt_parse = gt_dict.get("gt_parse", gt_dict)
        except (json.JSONDecodeError, TypeError):
            continue
        
        plot_type = gt_parse.get("panels", [{}])[0].get("plot_type", "unknown") if "panels" in gt_parse else "unknown"
        results["total"] += 1
        
        pred_text = run_inference(model, processor, device, image_path, "p2n", max_length)
        
        try:
            pred_json = json.loads(pred_text)
            results["json_valid"] += 1
            
            mae, rmse = compute_numerical_metrics(pred_json, gt_parse)
            if mae is not None:
                results["mae_values"].append(mae)
                results["rmse_values"].append(rmse)
                
                if plot_type not in results["per_type"]:
                    results["per_type"][plot_type] = {"mae": [], "count": 0}
                results["per_type"][plot_type]["mae"].append(mae)
                results["per_type"][plot_type]["count"] += 1
                
        except json.JSONDecodeError:
            pass
    
    return results


def evaluate_axis_info(model, processor, device, samples, max_length):
    """Evaluate axis detection accuracy."""
    results = {"total": 0, "scale_correct": 0, "label_correct": 0}
    
    for sample in samples:
        gt_text = sample["ground_truth"]
        image_path = sample["image_path"]
        
        try:
            gt_dict = json.loads(gt_text)
            gt_parse = gt_dict.get("gt_parse", {})
        except (json.JSONDecodeError, TypeError):
            continue
        
        if gt_parse.get("task_type") != "axis_info":
            continue
        
        results["total"] += 1
        pred_text = run_inference(model, processor, device, image_path, "axis_info", max_length)
        
        try:
            pred = json.loads(pred_text)
            if pred.get("x_scale") == gt_parse.get("x_scale") and pred.get("y_scale") == gt_parse.get("y_scale"):
                results["scale_correct"] += 1
            if pred.get("x_label", "").strip() == gt_parse.get("x_label", "").strip():
                results["label_correct"] += 1
        except json.JSONDecodeError:
            pass
    
    return results


def evaluate_element_count(model, processor, device, samples, max_length):
    """Evaluate element detection accuracy."""
    bool_fields = ["has_legend", "has_colorbar", "has_error_bars", "has_grid"]
    results = {"total": 0, "field_correct": {f: 0 for f in bool_fields}, "n_series_mae": []}
    
    for sample in samples:
        gt_text = sample["ground_truth"]
        image_path = sample["image_path"]
        
        try:
            gt_dict = json.loads(gt_text)
            gt_parse = gt_dict.get("gt_parse", {})
        except (json.JSONDecodeError, TypeError):
            continue
        
        if gt_parse.get("task_type") != "element_count":
            continue
        
        results["total"] += 1
        pred_text = run_inference(model, processor, device, image_path, "element_count", max_length)
        
        try:
            pred = json.loads(pred_text)
            for field in bool_fields:
                if pred.get(field) == gt_parse.get(field):
                    results["field_correct"][field] += 1
            
            if "n_series" in pred and "n_series" in gt_parse:
                results["n_series_mae"].append(abs(pred["n_series"] - gt_parse["n_series"]))
        except json.JSONDecodeError:
            pass
    
    return results


def evaluate(model_dir, dataset_path, num_samples, max_length):
    """Run full multi-task evaluation."""
    model, processor, device = load_model(model_dir)
    print(f"Model loaded on {device}")
    
    # Load metadata
    metadata_path = os.path.join(dataset_path, "metadata.jsonl")
    all_samples = []
    with open(metadata_path) as f:
        for line in f:
            entry = json.loads(line)
            image_path = os.path.join(dataset_path, entry["file_name"])
            if os.path.exists(image_path):
                all_samples.append({
                    "image_path": image_path,
                    "ground_truth": entry["ground_truth"],
                })
    
    print(f"Found {len(all_samples)} samples")
    
    # Split by task
    main_samples = [s for s in all_samples if '"task":' not in s["ground_truth"] or '"task": "p2n"' in s["ground_truth"]]
    axis_samples = [s for s in all_samples if '"task": "axis_info"' in s["ground_truth"]]
    elem_samples = [s for s in all_samples if '"task": "element_count"' in s["ground_truth"]]
    
    # Subsample
    random.seed(42)
    main_eval = random.sample(main_samples, min(num_samples, len(main_samples)))
    axis_eval = random.sample(axis_samples, min(num_samples, len(axis_samples)))
    elem_eval = random.sample(elem_samples, min(num_samples, len(elem_samples)))
    
    print(f"\nEvaluating: {len(main_eval)} main, {len(axis_eval)} axis, {len(elem_eval)} element\n")
    
    # === Task 1: Main P2N extraction ===
    print("=" * 60)
    print("TASK: Plot-to-Numbers (p2n)")
    print("=" * 60)
    p2n_results = evaluate_p2n(model, processor, device, main_eval, max_length)
    
    total = p2n_results["total"]
    valid = p2n_results["json_valid"]
    print(f"  JSON parse rate:   {valid}/{total} ({100*valid/max(total,1):.1f}%)")
    if p2n_results["mae_values"]:
        print(f"  Mean MAE:          {np.mean(p2n_results['mae_values']):.4f}")
        print(f"  Mean RMSE:         {np.mean(p2n_results['rmse_values']):.4f}")
        print(f"  Median MAE:        {np.median(p2n_results['mae_values']):.4f}")
    
    if p2n_results["per_type"]:
        print(f"\n  Per-type MAE (top 10 worst):")
        sorted_types = sorted(p2n_results["per_type"].items(), 
                            key=lambda x: np.mean(x[1]["mae"]), reverse=True)
        for pt, data in sorted_types[:10]:
            print(f"    {pt:30s} MAE={np.mean(data['mae']):.4f}  (n={data['count']})")
    
    # === Task 2: Axis info ===
    print(f"\n{'=' * 60}")
    print("TASK: Axis Info Detection")
    print("=" * 60)
    axis_results = evaluate_axis_info(model, processor, device, axis_eval, max_length)
    
    total = axis_results["total"]
    if total > 0:
        print(f"  Scale accuracy:    {axis_results['scale_correct']}/{total} ({100*axis_results['scale_correct']/total:.1f}%)")
        print(f"  Label accuracy:    {axis_results['label_correct']}/{total} ({100*axis_results['label_correct']/total:.1f}%)")
    else:
        print("  No axis_info samples found.")
    
    # === Task 3: Element count ===
    print(f"\n{'=' * 60}")
    print("TASK: Element Count Detection")
    print("=" * 60)
    elem_results = evaluate_element_count(model, processor, device, elem_eval, max_length)
    
    total = elem_results["total"]
    if total > 0:
        for field, correct in elem_results["field_correct"].items():
            print(f"  {field:20s} accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        if elem_results["n_series_mae"]:
            print(f"  n_series MAE:      {np.mean(elem_results['n_series_mae']):.2f}")
    else:
        print("  No element_count samples found.")
    
    # === Summary ===
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    summary = {
        "p2n_json_parse_rate": round(100 * p2n_results["json_valid"] / max(p2n_results["total"], 1), 1),
        "p2n_mean_mae": round(float(np.mean(p2n_results["mae_values"])), 4) if p2n_results["mae_values"] else None,
        "p2n_mean_rmse": round(float(np.mean(p2n_results["rmse_values"])), 4) if p2n_results["rmse_values"] else None,
        "axis_scale_accuracy": round(100 * axis_results["scale_correct"] / max(axis_results["total"], 1), 1),
        "element_count_total": elem_results["total"],
    }
    
    # Save results
    if os.path.exists(model_dir):
        results_path = os.path.join(model_dir, "eval_results.json")
    else:
        results_path = "eval_results.json"
        
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to {results_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate P2N model across all tasks.")
    parser.add_argument("--model_dir", type=str, default="./p2n-model", help="Path to trained model")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples per task to evaluate")
    parser.add_argument("--max_length", type=int, default=1536, help="Max decoder sequence length")
    args = parser.parse_args()
    
    evaluate(args.model_dir, args.dataset_path, args.num_samples, args.max_length)
