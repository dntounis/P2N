import json
import torch
import argparse
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
import glob
import os

def evaluate(model_dir, dataset_path):
    print(f"Loading model from {model_dir}")
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Actually on mac we might want "mps"
    if torch.backends.mps.is_available():
        device = "mps"
    model.to(device)
    model.eval()

    images_path = os.path.join(dataset_path, "images", "*.png")
    image_files = sorted(glob.glob(images_path))
    
    if not image_files:
        print("No images found for evaluation.")
        return

    # Evaluate on the first image for demonstration
    test_image_path = image_files[0]
    print(f"Evaluating on {test_image_path}")
    image = Image.open(test_image_path).convert("RGB")
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Prepare decoder inputs
    task_prompt = "<s_p2n>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    
    # Strip prompt
    if sequence.startswith(task_prompt):
        sequence = sequence[len(task_prompt):]
        
    print("\n--- Model Output ---")
    print(sequence)
    print("--------------------")
    
    try:
        parsed_json = json.loads(sequence)
        print("Successfully parsed generated text as JSON.")
    except json.JSONDecodeError:
        print("Output is not valid JSON.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate P2N model.")
    parser.add_argument("--model_dir", type=str, default="./p2n-model", help="Path to trained model")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset")
    args = parser.parse_args()
    
    evaluate(args.model_dir, args.dataset_path)
