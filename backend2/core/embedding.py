# backend2/core/embedding.py
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class ElementEmbedder:
    """Handles the embedding of UI elements using transformer models"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.feature_extractor = UIFeatureExtractor()

    def encode_elements(self, elements: List[Dict]) -> np.ndarray:
        """Create embeddings for UI elements with feature fusion"""
        # Get text descriptions
        descriptions = [self._format_element_text(elem) for elem in elements]

        # Get semantic embeddings from transformer
        semantic_embeddings = self.model.encode(descriptions)

        # Extract numerical features
        numerical_features = np.array([
            self.feature_extractor.extract_features(elem) for elem in elements
        ])

        return semantic_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query)

    def _format_element_text(self, elem: Dict) -> str:
        """Format element properties into descriptive text"""
        description = []

        # Element type and ID
        elem_type = elem.get("type", "unknown").lower()
        elem_id = elem.get("id", "unknown")
        description.append(f"UI element of type {elem_type} with ID {elem_id}")

        # Text content
        if text := elem.get("text", ""):
            description.append(f"displaying text '{text}'")

        # Description content
        if desc := elem.get("content_desc", ""):
            description.append(f"with description '{desc}'")

        # Interactivity
        if elem.get("clickable", False):
            description.append("can be clicked")

        if elem.get("enabled", True):
            description.append("is enabled")
        else:
            description.append("is disabled")

        # Location information
        bounds = elem.get("bounds", [])
        if len(bounds) >= 4:
            x1, y1, x2, y2 = bounds[:4]
            position = self._describe_position(x1, y1, x2, y2)
            description.append(f"located at the {position} of the screen")

        return " ".join(description)

    def _describe_position(self, x1, y1, x2, y2, screen_w=1440, screen_h=2560) -> str:
        """Convert coordinates to human-readable position description"""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Vertical position
        if center_y < screen_h * 0.33:
            v_pos = "top"
        elif center_y < screen_h * 0.66:
            v_pos = "middle"
        else:
            v_pos = "bottom"

        # Horizontal position
        if center_x < screen_w * 0.33:
            h_pos = "left"
        elif center_x < screen_w * 0.66:
            h_pos = "center"
        else:
            h_pos = "right"

        return f"{v_pos} {h_pos}"


class UIFeatureExtractor:
    """Extracts numerical features from UI elements for machine learning"""

    def extract_features(self, elem: Dict) -> np.ndarray:
        """Extract numerical features from a UI element"""
        features = []

        # 1. Position and size features
        bounds = elem.get("bounds", [0, 0, 0, 0])
        if len(bounds) < 4:
            bounds = bounds + [0] * (4 - len(bounds))

        x1, y1, x2, y2 = bounds
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        area = width * height
        aspect_ratio = width / max(height, 1)

        screen_w, screen_h = 1440, 2560
        norm_x1, norm_y1 = x1 / screen_w, y1 / screen_h
        norm_x2, norm_y2 = x2 / screen_w, y2 / screen_h
        norm_width = width / screen_w
        norm_height = height / screen_h
        norm_area = norm_width * norm_height

        center_x = (norm_x1 + norm_x2) / 2
        center_y = (norm_y1 + norm_y2) / 2

        position_features = [
            norm_x1, norm_y1, norm_x2, norm_y2,
            norm_width, norm_height, norm_area, aspect_ratio,
            center_x, center_y
        ]
        features.extend(position_features)

        # 2. Element properties as binary features
        property_features = [
            1.0 if elem.get("clickable", False) else 0.0,
            1.0 if elem.get("enabled", True) else 0.0,
            1.0 if elem.get("text", "") else 0.0,
            1.0 if elem.get("content_desc", "") else 0.0,
            min(elem.get("depth", 0) / 10.0, 1.0),  # Normalized depth
        ]
        features.extend(property_features)

        text = elem.get("text", "")
        text_features = [
            min(len(text) / 50.0, 1.0),  # Normalized text length
            1.0 if any(c.isupper() for c in text) else 0.0,  # Has uppercase
            1.0 if any(c.isdigit() for c in text) else 0.0,  # Has digits
        ]
        features.extend(text_features)

        element_types = [
            "button", "text", "input", "image", "view",
            "checkbox", "radio", "dropdown", "container"
        ]
        elem_type = elem.get("type", "").lower()
        type_features = [1.0 if t in elem_type else 0.0 for t in element_types]
        features.extend(type_features)

        return np.array(features, dtype=np.float32)
