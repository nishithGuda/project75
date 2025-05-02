# backend2/core/fusion_classifier.py
import time
from typing import Any, Dict, List
import torch
import numpy as np
from .navigator import UINavigator
from models.llm_model_bert import LLMQueryElementClassifier
from transformers import BertTokenizer


class HybridNavigator(UINavigator):
    """Enhanced navigator that combines BERT, RAG and RL approaches"""

    def __init__(self, *args,
                 bert_model_path="model/llm_bert_model.pt",
                 rl_model_path="model/rl_model.pt",
                 element_dim=20,
                 **kwargs):
        # Initialize the base navigator with RL model
        super().__init__(*args, rl_model_path=rl_model_path, **kwargs)

        self.element_dim = element_dim

        # Initialize BERT components
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = LLMQueryElementClassifier(
            element_feature_dim=element_dim)

        # Load the model and move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.load_state_dict(
            torch.load(bert_model_path, map_location=device))
        self.bert_model.to(device)
        self.bert_model.eval()

        self.device = device
        print(f"BERT model loaded from {bert_model_path}")
        print(f"Complete hybrid system initialized with BERT, RAG and RL components")

    def process_query(self, query: str, screen_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Override process_query to include BERT analysis"""
        start_time = time.time()
        self.metrics["total_queries"] += 1

        try:
            # Extract elements from metadata
            elements = screen_metadata.get("elements", [])
            if not elements:
                return {
                    "success": False,
                    "error": "No UI elements found in screen metadata",
                    "actions": []
                }

            # 1. Vector-based retrieval
            vector_candidates = self._retrieve_similar_elements(
                query, elements)

            if not vector_candidates:
                return {
                    "success": False,
                    "error": "No relevant elements found for this query",
                    "actions": []
                }

            # 2. Add action type prediction using embedder
            candidates_with_actions = self._add_action_prediction(
                query, vector_candidates)

            # 3. LLM-based analysis
            candidates_with_llm = self._analyze_with_llm(
                query, candidates_with_actions, screen_metadata)

            # 4. Calculate final confidence scores with BERT
            actions = self._calculate_confidence_scores(
                candidates_with_llm, query, screen_metadata)

            # 5. Update performance metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time)

            # Extract base action for the query
            base_action = self._extract_base_action_type(query)

            return {
                "success": True,
                "actions": actions,
                "query": query,
                "detected_action": base_action,
                "processing_time": round(processing_time, 3),
                "vector_candidates": len(vector_candidates),
                "total_elements": len(elements)
            }

        except Exception as e:
            import traceback
            traceback.print_exc()

            processing_time = time.time() - start_time
            return {
                "success": False,
                "error": f"Error processing query: {str(e)}",
                "actions": [],
                "processing_time": round(processing_time, 3)
            }

    def _calculate_confidence_scores(self, elements: List[Dict],
                                     query: str, screen_metadata: Dict) -> List[Dict]:
        """Calculate final confidence scores using BERT, RAG and RL"""
        actions = []
        screen_id = screen_metadata.get("screen_id", "unknown")

        # Get BERT predictions for all elements
        bert_scores = self._get_bert_scores(query, elements)

        for elem in elements:
            # Get scores from different sources
            vector_score = elem.get("vector_score", 0.5)
            llm_score = elem.get("llm_confidence", 0.5)
            action_confidence = elem.get("action_confidence", 0.7)

            # Get element ID
            element_id = elem.get("id", "unknown")

            # Get BERT score
            bert_score = bert_scores.get(element_id, 0.5)

            # Get historical success rate if available
            history_key = f"{screen_id}_{element_id}"
            history_score = 0.5
            if history_key in self.history:
                history = self.history[history_key]
                interactions = history.get("interactions", 0)
                successes = history.get("successes", 0)

                if interactions > 0:
                    # Calculate success rate with a prior
                    history_score = (successes + 1) / (interactions + 2)

            # Apply weighting to combine scores
            # Distribute weights among all components
            llm_weight = 0.30
            vector_weight = 0.10
            bert_weight = 0.30
            action_weight = 0.15
            history_weight = 0.15

            combined_score = (
                llm_weight * llm_score +
                vector_weight * vector_score +
                bert_weight * bert_score +
                action_weight * action_confidence +
                history_weight * history_score
            )

            # Apply RL model adjustment
            if self.rl_model is not None:
                try:
                    with torch.no_grad():
                        # Get model device
                        model_device = next(self.rl_model.parameters()).device

                        # For the existing model architecture, we only use the combined score
                        features = torch.tensor(
                            [[combined_score]], dtype=torch.float).to(model_device)

                        # Get RL adjustment
                        adjustment = self.rl_model(features)

                        # Move result to CPU before calling item()
                        adjustment_value = adjustment.cpu().item()

                        # Blend original score with RL adjustment (70/30 blend)
                        combined_score = 0.7 * combined_score + 0.3 * adjustment_value
                except Exception as e:
                    print(f"Error applying RL model: {e}")

            # Get action type from element or default to click
            action_type = elem.get("action_type", "click")

            # Get action parameters if available
            action_params = elem.get("action_parameters", None)

            # Create action object
            action = {
                "id": element_id,
                "type": elem.get("type", "unknown"),
                "text": elem.get("text", ""),
                "confidence": round(float(combined_score), 4),
                "action_type": action_type,
                "reasoning": elem.get("reasoning", ""),
                # Include individual scores for transparency
                "llm_score": round(float(llm_score), 4),
                "vector_score": round(float(vector_score), 4),
                "bert_score": round(float(bert_score), 4),
                "action_confidence": round(float(action_confidence), 4),
                "history_score": round(float(history_score), 4)
            }

            # Add action parameters if available
            if action_params:
                action["action_parameters"] = action_params

            actions.append(action)

        # Sort by confidence
        actions.sort(key=lambda x: x["confidence"], reverse=True)

        return actions

    def _get_bert_scores(self, query, elements):
        """Get confidence scores from BERT model"""
        scores = {}

        for element in elements:
            # Extract features that match the model's expected input
            element_features = self._extract_bert_features(element)
            element_feat = torch.tensor(
                element_features, dtype=torch.float).unsqueeze(0).to(self.device)

            # Tokenize the query
            encoding = self.tokenizer(
                query,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )

            # Move tensors to device
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Get prediction
            with torch.no_grad():
                try:
                    logits = self.bert_model(
                        query_input_ids=input_ids,
                        query_attention_mask=attention_mask,
                        element_features=element_feat
                    )
                    # Convert to probability
                    prob = torch.sigmoid(logits).cpu().item()

                    # Store score
                    element_id = element.get("id", "unknown")
                    scores[element_id] = prob
                except Exception as e:
                    print(f"Error getting BERT prediction: {e}")
                    # Default score
                    element_id = element.get("id", "unknown")
                    scores[element_id] = 0.5

        return scores

    def _extract_bert_features(self, elem):
        """Extract features for BERT model"""
        # Implementation copied from inference_bert.py
        features = []
        screen_width, screen_height = 1440, 2560

        # 1. Position and size (normalized) - 6 features
        bounds = elem.get("bounds", [0, 0, 0, 0])
        if len(bounds) < 4:
            bounds = bounds + [0] * (4 - len(bounds))

        x1, y1, x2, y2 = bounds

        # Normalized position and size
        center_x = ((x1 + x2) / 2) / screen_width
        center_y = ((y1 + y2) / 2) / screen_height
        width = (x2 - x1) / screen_width
        height = (y2 - y1) / screen_height
        area = width * height
        aspect_ratio = width / max(height, 0.001)

        position_features = [center_x, center_y,
                             width, height, area, aspect_ratio]
        features.extend(position_features)

        # 2. Element properties - 4 features
        clickable = 1.0 if elem.get("clickable", False) else 0.0
        enabled = 1.0 if elem.get("enabled", True) else 0.0
        visible = 1.0 if elem.get("visible", True) else 0.0
        depth = min(elem.get("depth", 0) / 10.0, 1.0)

        features.extend([clickable, enabled, visible, depth])

        # 3. Text presence - 2 features
        has_text = 1.0 if elem.get("text", "") else 0.0
        has_desc = 1.0 if elem.get("content_desc", "") else 0.0

        features.extend([has_text, has_desc])

        # 4. Element type - 8 features (simplified)
        known_types = [
            "button", "text", "input", "image",
            "checkbox", "radio", "dropdown", "unknown"
        ]

        type_vec = [0.0] * len(known_types)
        elem_type = elem.get("type", "unknown").lower()

        if elem_type in known_types:
            type_vec[known_types.index(elem_type)] = 1.0
        else:
            type_vec[-1] = 1.0  # Mark as unknown

        features.extend(type_vec)  # 8 features

        # Ensure we have exactly 20 features
        if len(features) < self.element_dim:
            features.extend([0.0] * (self.element_dim - len(features)))

        return features[:self.element_dim]
