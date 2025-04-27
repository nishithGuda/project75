import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import json
import os
import numpy as np


class LLMQueryElementDataset(Dataset):
    def __init__(self, json_path, max_elements=10, element_dim=20, max_len=64):
        self.samples = []
        self.max_elements = max_elements
        self.element_dim = element_dim
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Define required feature size to validate element_dim
        self.known_types = ["button", "text", "input", "image", "dropdown", "checkbox", "radio", "view", "layout", "unknown"]
        self.min_feature_size = 4 + 3 + len(self.known_types)  # bounds + [clickable, enabled, text_len] + one_hot_type
        
        if element_dim < self.min_feature_size:
            raise ValueError(f"element_dim must be at least {self.min_feature_size}")

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            query = entry["query"]
            elements = entry.get("elements", [])[:max_elements]
            target_idx = entry.get("target_idx", -1)

            if target_idx >= 0 and target_idx < len(elements):
                for i, elem in enumerate(elements):
                    label = 1.0 if i == target_idx else 0.0
                    self.samples.append((query, elem, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        query, elem, label = self.samples[index]

        encoding = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        element_vector = self.encode_element(elem)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "element_features": torch.tensor(element_vector, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float)      
        }
    
    def encode_element(self, elem):
        # Extract features with defensive coding
        bounds = elem.get("bounds", [0, 0, 0, 0])
        if len(bounds) < 4:
            bounds = bounds + [0] * (4 - len(bounds))  # Ensure at least 4 values
            
        clickable = 1.0 if elem.get("clickable", False) else 0.0
        enabled = 1.0 if elem.get("enabled", False) else 0.0
        text = elem.get("text", "")
        text_len = min(len(text) / 50.0, 1.0)  # Normalize and cap at 1.0

        type_vec = self.one_hot_type(elem.get("type", "unknown"))
        features = bounds[:4] + [clickable, enabled, text_len] + type_vec

        # Pad or truncate to element_dim
        if len(features) < self.element_dim:
            features = features + [0.0] * (self.element_dim - len(features))
        return features[:self.element_dim]
    
    def one_hot_type(self, t):
        vec = [0.0] * len(self.known_types)

        t = t.lower()
        if t in self.known_types:
            vec[self.known_types.index(t)] = 1.0
        else:
            vec[-1] = 1.0  # Mark as unknown
        return vec