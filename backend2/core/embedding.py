# backend2/core/embedding.py
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional


class ElementEmbedder:
    """Handles the embedding of UI elements using transformer models"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.feature_extractor = UIFeatureExtractor()

        # Action type prediction components
        self.action_types = {
            "click": "Click or tap on an element",
            "input": "Type, enter or fill in text in a field",
            "select": "Choose an option from a dropdown or list",
            "view": "View or display information",
            "transfer": "Transfer money or funds between accounts",
            "search": "Search for something specific",
            "filter": "Filter or sort information",
            "navigate": "Navigate to a different page or section"
        }

        # Patterns for action type detection
        self.action_patterns = [
            (r'\b(click|tap|press|select|choose)\b', 'click'),
            (r'\b(type|enter|input|fill|write)\b', 'input'),
            (r'\b(scroll|swipe)\b', 'scroll'),
            (r'\b(view|show|display|see|look at)\b', 'view'),
            (r'\b(transfer|send|move)\s+(money|funds|balance|cash)\b', 'transfer'),
            (r'\b(search|find|look for)\b', 'search'),
            (r'\b(filter|sort)\b', 'filter'),
            (r'\b(navigate|go to|open|visit)\b', 'navigate'),
        ]

        # Create action type embeddings
        self.action_embeddings = self._create_action_embeddings()

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

    def extract_action_from_query(self, query: str) -> str:
        """Extract action type from query using regex patterns"""
        query_lower = query.lower()

        # Try pattern matching first
        for pattern, action in self.action_patterns:
            if re.search(pattern, query_lower):
                return action

        # Default to click if no pattern matches
        return "click"

    def extract_input_value(self, query: str) -> Optional[str]:
        """Extract input value from query for input actions"""
        # Only extract value for input actions
        if not re.search(r'\b(type|enter|input|fill|write)\b', query.lower()):
            return None

        # Patterns to extract quoted text or text after specific phrases
        patterns = [
            # Match quoted text
            r'(?:type|enter|input|fill|write)\s+[\'"]([^\'"]+)[\'"]',
            # Match unquoted text
            r'(?:type|enter|input|fill|write)\s+(?:the\s+)?(?:text|value|words?)?\s*:?\s*([a-zA-Z0-9\s]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip()

        return None

    def predict_action_type(self, query: str, element: Optional[Dict] = None) -> Tuple[str, float]:
        """Predict the most likely action type for a query and element"""
        # 1. Try regex pattern matching first (high precision)
        query_lower = query.lower()
        for pattern, action in self.action_patterns:
            if re.search(pattern, query_lower):
                return action, 0.9  # High confidence for regex matches

        # 2. Use embedding similarity if regex doesn't match
        query_embedding = self.encode_query(query)

        # Calculate similarity with each action type
        similarities = {}
        for action, embedding in self.action_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities[action] = similarity

        # 3. Consider element type if available
        if element is not None:
            element_type = element.get("type", "").lower()

            # Boost certain action types based on element type
            if element_type in ["input", "textarea"]:
                similarities["input"] = max(similarities["input"], 0.7)
            elif element_type in ["select", "dropdown"]:
                similarities["select"] = max(similarities["select"], 0.7)
            elif element_type == "button":
                similarities["click"] = max(similarities["click"], 0.7)

        # Get action with highest similarity
        best_action = max(similarities, key=similarities.get)
        confidence = similarities[best_action]

        # Default to click if confidence is too low
        if confidence < 0.4:
            return "click", 0.5

        return best_action, confidence

    def predict_actions_for_elements(self, query: str, elements: List[Dict]) -> List[Dict]:
        """Predict action types for multiple elements with the same query"""
        # Extract base action from query
        default_action = self.extract_action_from_query(query)
        input_value = self.extract_input_value(
            query) if default_action == "input" else None

        # Process each element
        updated_elements = []
        for elem in elements:
            elem_copy = elem.copy()

            # Get element-specific prediction
            action, confidence = self.predict_action_type(query, elem)

            # Add action information to the element
            elem_copy["action_type"] = action
            elem_copy["action_confidence"] = confidence

            # Add action parameters for input actions if needed
            if action == "input" and input_value:
                elem_copy["action_parameters"] = {"value": input_value}

            updated_elements.append(elem_copy)

        return updated_elements

    def _create_action_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings for each action type"""
        embeddings = {}

        # Create rich descriptions for each action type
        action_descriptions = {}
        for action, desc in self.action_types.items():
            # Create multiple descriptions for each action
            descriptions = [
                desc,
                f"I want to {action}",
                f"The user wants to {action}",
                f"{action} something"
            ]
            action_descriptions[action] = descriptions

        # Create embeddings for all descriptions
        for action, descriptions in action_descriptions.items():
            # Get embeddings for each description
            desc_embeddings = self.model.encode(descriptions)

            # Average the embeddings for each action type
            embeddings[action] = np.mean(desc_embeddings, axis=0)

        return embeddings

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
