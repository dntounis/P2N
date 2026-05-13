"""P2N Inference — Run trained model on new plot images.

Supports all three tasks:
  - p2n: Extract full structured data from a plot
  - axis_info: Detect axis scales, labels, orientation
  - element_count: Detect structural elements (legend, grid, error bars, etc.)

Usage:
    # Single image, main task
    python inference.py --model_dir ./p2n-model --image plot.png

    # Single image, specific task
    python inference.py --model_dir ./p2n-model --image plot.png --task axis_info

    # Batch mode on a directory
    python inference.py --model_dir ./p2n-model --image_dir ./test_images/ --task p2n

    # All tasks on one image
    python inference.py --model_dir ./p2n-model --image plot.png --task all
"""
import json
import os
import glob
import argparse
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
        print("Model loaded with accelerate (multi-GPU supported).")
    except ImportError:
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        model.to(device)
        print(f"Model loaded on {device} (single device). For multi-GPU, 'pip install accelerate'.")
        
    model.eval()
    return model, processor, device


def infer(model, processor, device, image_path, task="p2n", max_length=1536):
    """Run inference on a single image for a given task.
    
    Args:
        model: The trained VisionEncoderDecoder model
        processor: The DonutProcessor
        device: torch device
        image_path: Path to the plot image
        task: One of 'p2n', 'axis_info', 'element_count'
        max_length: Maximum decoder sequence length
    
    Returns:
        dict: Parsed JSON output, or raw string if JSON parsing fails
    """
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    start_tok, end_tok = TASK_TOKENS.get(task, TASK_TOKENS["p2n"])
    decoder_input_ids = processor.tokenizer(
        start_tok, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    
    # Move inputs to the same device as the model's first layer
    input_device = next(model.parameters()).device
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
    # Clean up all special tokens
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")
    for tok_pair in TASK_TOKENS.values():
        for tok in tok_pair:
            sequence = sequence.replace(tok, "")
    sequence = sequence.strip()
    
    try:
        return json.loads(sequence)
    except json.JSONDecodeError:
        return {"raw_output": sequence, "parse_error": True}


def run_all_tasks(model, processor, device, image_path, max_length=1536):
    """Run all three tasks on a single image."""
    results = {}
    for task in TASK_TOKENS:
        results[task] = infer(model, processor, device, image_path, task, max_length)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run P2N inference on plot images.")
    parser.add_argument("--model_dir", type=str, default="./p2n-model", help="Path to trained model")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to a directory of images")
    parser.add_argument("--task", type=str, default="p2n",
                        choices=["p2n", "axis_info", "element_count", "all"],
                        help="Task to run")
    parser.add_argument("--max_length", type=int, default=1536, help="Max decoder sequence length")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Must specify either --image or --image_dir")
    
    model, processor, device = load_model(args.model_dir)
    
    # Collect images
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            glob.glob(os.path.join(args.image_dir, "*.png")) +
            glob.glob(os.path.join(args.image_dir, "*.jpg")) +
            glob.glob(os.path.join(args.image_dir, "*.jpeg"))
        )
    
    print(f"Processing {len(image_paths)} image(s) with task='{args.task}'")
    
    all_results = {}
    for img_path in image_paths:
        print(f"\n{'─' * 50}")
        print(f"Image: {img_path}")
        
        if args.task == "all":
            result = run_all_tasks(model, processor, device, img_path, args.max_length)
            for task_name, task_result in result.items():
                print(f"\n  [{task_name}]:")
                print(f"  {json.dumps(task_result, indent=2)}")
        else:
            result = infer(model, processor, device, img_path, args.task, args.max_length)
            print(json.dumps(result, indent=2))
        
        all_results[os.path.basename(img_path)] = result
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
