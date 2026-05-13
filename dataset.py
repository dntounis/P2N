"""P2N Dataset loader with multi-task support.

Supports three task types:
  - p2n (main): Full plot-to-numbers extraction
  - axis_info: Axis scale, label, and orientation detection
  - element_count: Structural element detection (legend, grid, error bars, etc.)
"""
import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# Task token mapping
TASK_TOKENS = {
    "p2n":           ("<s_p2n>",           "</s_p2n>"),
    "axis_info":     ("<s_axis_info>",     "</s_axis_info>"),
    "element_count": ("<s_element_count>", "</s_element_count>"),
}

ALL_SPECIAL_TOKENS = [tok for pair in TASK_TOKENS.values() for tok in pair]


class P2NDataset(Dataset):
    def __init__(self, dataset_path, processor, split="train", max_length=1536):
        self.processor = processor
        self.max_length = max_length
        
        # Use Hugging Face imagefolder loader
        self.dataset = load_dataset("imagefolder", data_dir=dataset_path, split=split)
        
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
