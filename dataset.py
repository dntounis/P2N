import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class P2NDataset(Dataset):
    def __init__(self, dataset_path, processor, split="train", max_length=512):
        self.processor = processor
        self.max_length = max_length
        
        # We use Hugging Face datasets to load the imagefolder
        self.dataset = load_dataset("imagefolder", data_dir=dataset_path, split=split)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        
        # In donut, ground_truth is a string, but the imagefolder loader parses it if it's jsonl.
        # Wait, the dataset loader returns it based on the jsonl. 
        # In our metadata.jsonl, "ground_truth" is a string.
        gt_string = item['ground_truth']
        
        # Donut expects the task start token to be added
        task_prompt = "<s_p2n>"
        
        # Format the text
        # E.g. <s_p2n>{"data": [{"x": 10.0, "y": 20.0}]}</s_p2n>
        # We parse the gt_string to just get the dict, then convert to our standard format.
        gt_dict = json.loads(gt_string)
        sequence = task_prompt + json.dumps(gt_dict['gt_parse']) + self.processor.tokenizer.eos_token
        
        # Prepare inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        
        # Tokenize labels
        labels = self.processor.tokenizer(
            sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}
