"""P2N Dataset loader with multi-task support.

Supports three task types:
  - p2n (main): Full plot-to-numbers extraction
  - axis_info: Axis scale, label, and orientation detection
  - element_count: Structural element detection (legend, grid, error bars, etc.)
"""
import json
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image

# Task token mapping
TASK_TOKENS = {
    "p2n":           ("<s_p2n>",           "</s_p2n>"),
    "axis_info":     ("<s_axis_info>",     "</s_axis_info>"),
    "element_count": ("<s_element_count>", "</s_element_count>"),
}

ALL_SPECIAL_TOKENS = [tok for pair in TASK_TOKENS.values() for tok in pair]


def load_p2n_imagefolder(dataset_path, split=None):
    """Load generated P2N data in HF imagefolder metadata.jsonl format."""
    metadata_path = os.path.join(dataset_path, "metadata.jsonl")
    if os.path.exists(metadata_path):
        return load_dataset("imagefolder", data_files=metadata_path, split=split)
    return load_dataset("imagefolder", data_dir=dataset_path, split=split)


class P2NMetadataDataset(Dataset):
    """Read metadata.jsonl directly and preprocess examples on demand."""

    def __init__(
        self,
        dataset_path,
        processor,
        max_length=512,
        use_torchvision_decode=False,
        cuda_jpeg_decode=False,
    ):
        self.dataset_path = dataset_path
        self.processor = processor
        self.max_length = max_length
        self.use_torchvision_decode = use_torchvision_decode
        self.cuda_jpeg_decode = cuda_jpeg_decode
        metadata_path = os.path.join(dataset_path, "metadata.jsonl")

        self.entries = []
        with open(metadata_path) as f:
            for line in f:
                entry = json.loads(line)
                image_path = os.path.join(dataset_path, entry["file_name"])
                if os.path.exists(image_path):
                    self.entries.append((image_path, entry["ground_truth"]))

    def __len__(self):
        return len(self.entries)

    def _load_image(self, image_path):
        if self.use_torchvision_decode and image_path.lower().endswith((".jpg", ".jpeg")):
            try:
                from torchvision.io import ImageReadMode, decode_jpeg, read_file

                encoded = read_file(image_path)
                device = "cuda" if self.cuda_jpeg_decode and torch.cuda.is_available() else "cpu"
                image = decode_jpeg(encoded, mode=ImageReadMode.RGB, device=device)
                return image.cpu().permute(1, 2, 0).numpy()
            except Exception:
                pass

        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        image_path, gt_text = self.entries[idx]
        image = self._load_image(image_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        try:
            gt_dict = json.loads(gt_text)
            task = gt_dict.get("task", "p2n")
        except (json.JSONDecodeError, TypeError):
            task = "p2n"

        start_tok, end_tok = TASK_TOKENS.get(task, TASK_TOKENS["p2n"])
        target_text = f"{start_tok}{gt_text}{end_tok}"

        labels = self.processor.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


class P2NDataset(Dataset):
    def __init__(self, dataset_path, processor, split="train", max_length=1536):
        self.processor = processor
        self.max_length = max_length
        
        # Use Hugging Face imagefolder loader
        self.dataset = load_p2n_imagefolder(dataset_path, split=split)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        
        gt_string = item['ground_truth']
        
        # Determine task type from ground truth
        try:
            gt_dict = json.loads(gt_string)
            task = gt_dict.get("task", "p2n")
            gt_parse = gt_dict.get("gt_parse", gt_dict)
        except (json.JSONDecodeError, TypeError):
            task = "p2n"
            gt_parse = {}
        
        # Select the correct prompt tokens
        start_tok, end_tok = TASK_TOKENS.get(task, TASK_TOKENS["p2n"])
        sequence = start_tok + json.dumps(gt_parse) + end_tok
        
        # Prepare image inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Tokenize labels
        labels = self.processor.tokenizer(
            sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()
        
        # Replace padding token id's by -100 so they're ignored by the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}
