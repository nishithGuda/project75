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
        self.known_types = ["button", "text", "input", "image",
                            "dropdown", "checkbox", "radio", "view", "layout", "unknown"]
        # bounds + [clickable, enabled, text_len] + one_hot_type
        self.min_feature_size = 4 + 3 + len(self.known_types)

        if element_dim < self.min_feature_size:
            raise ValueError(
                f"element_dim must be at least {self.min_feature_size}")

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
        element_vector = self.enhanced_element_features(elem)

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
            # Ensure at least 4 values
            bounds = bounds + [0] * (4 - len(bounds))

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

    def enhanced_element_features(elem, screen_width=1440, screen_height=2560):
        """
        Extract richer features from UI elements

        Args:
            elem: Dictionary containing element properties
            screen_width: Width of the screen for normalization
            screen_height: Height of the screen for normalization

        Returns:
            feature_vector: List of features
        """
        features = []

        # 1. Position and size (normalized)
        bounds = elem.get("bounds", [0, 0, 0, 0])
        if len(bounds) < 4:
            bounds = bounds + [0] * (4 - len(bounds))

        # Raw position
        x1, y1, x2, y2 = bounds

        # Normalized coordinates
        norm_x1 = x1 / screen_width
        norm_y1 = y1 / screen_height
        norm_x2 = x2 / screen_width
        norm_y2 = y2 / screen_height

        # Center position
        center_x = (norm_x1 + norm_x2) / 2
        center_y = (norm_y1 + norm_y2) / 2

        # Size metrics
        width = norm_x2 - norm_x1
        height = norm_y2 - norm_y1
        area = width * height
        aspect_ratio = width / max(height, 0.001)  # Avoid division by zero

        # Position features
        position_features = [
            norm_x1, norm_y1, norm_x2, norm_y2,  # Boundaries
            center_x, center_y,                   # Center
            width, height, area, aspect_ratio     # Size metrics
        ]
        features.extend(position_features)

        # 2. Element properties
        prop_features = [
            1.0 if elem.get("clickable", False) else 0.0,
            1.0 if elem.get("enabled", True) else 0.0,
            1.0 if elem.get("visible", True) else 0.0,
            min(elem.get("depth", 0) / 10.0, 1.0),  # Normalized depth
            1.0 if elem.get("text", "") else 0.0,   # Has text
            1.0 if elem.get("content_desc", "") else 0.0  # Has description
        ]
        features.extend(prop_features)

        # 3. Text features
        text = elem.get("text", "")
        content_desc = elem.get("content_desc", "")

        # Text length (normalized)
        text_len = min(len(text) / 50.0, 1.0)
        desc_len = min(len(content_desc) / 50.0, 1.0)

        # Text characteristics
        has_uppercase = 1.0 if any(c.isupper() for c in text) else 0.0
        has_digit = 1.0 if any(c.isdigit() for c in text) else 0.0
        has_special_char = 1.0 if any(not c.isalnum() for c in text) else 0.0

        text_features = [
            text_len, desc_len,
            has_uppercase, has_digit, has_special_char
        ]
        features.extend(text_features)

        # 4. Element type (one-hot encoding)
        known_types = [
            "button", "text", "input", "image", "dropdown",
            "checkbox", "radio", "switch", "webview", "view", "layout", "unknown"
        ]

        type_vec = [0.0] * len(known_types)
        elem_type = elem.get("type", "unknown").lower()

        if elem_type in known_types:
            type_vec[known_types.index(elem_type)] = 1.0
        else:
            type_vec[-1] = 1.0  # Mark as unknown

        features.extend(type_vec)

        return features
