# backend2/core/navigator.py
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
import json
import os
import torch
import torch.nn as nn
from huggingface_hub import InferenceClient

from .vectorstore import VectorStore
from .embedding import ElementEmbedder


class MistralHFConnector:
    """Connector for Mistral LLM using Hugging Face Inference API"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                 provider: str = "nebius"):
        # Get API key from environment or parameter
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        if not self.api_key:
            raise ValueError(
                "HF API key is required. Set HF_API_KEY environment variable.")

        self.model = model
        self.provider = provider
        self.client = InferenceClient(
            provider=provider,
            api_key=self.api_key
        )
        self.cache = {}  # Simple cache for repeated queries

    def analyze_elements(self, query: str, elements: List[Dict],
                         screen_context: Optional[Dict] = None) -> List[Dict]:
        """Analyze UI elements against a user query using the LLM"""
        cache_key = f"{query}:{','.join([e.get('id', str(i)) for i, e in enumerate(elements)])}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Prepare system prompt
        system_message = {
            "role": "system",
            "content": """You are an AI expert in UI navigation, helping find the most relevant UI element for a user's query.

            Given a user's natural language query and UI element details:
            1. Analyze the semantic match between the query intent and each element's properties
            2. Consider element type, text content, position, and functionality
            3. Identify the action the user wants to perform (click, input, select, view, etc.)
            4. Rank elements by relevance, providing a confidence score (0-1) and brief reasoning

            Use a step-by-step approach to evaluate each element and determine the best match."""
        }

        # Prepare user prompt with element details
        user_content = f"User Query: \"{query}\"\n\n"

        # Add screen context if available
        if screen_context:
            user_content += "Current Screen Context:\n"
            user_content += f"- Screen ID: {screen_context.get('screen_id', 'unknown')}\n"
            user_content += f"- Path: {screen_context.get('path', 'unknown')}\n\n"

        # Add element descriptions
        user_content += "Available UI Elements:\n\n"

        for i, elem in enumerate(elements):
            elem_id = elem.get("id", f"element_{i}")
            elem_type = elem.get("type", "unknown")
            elem_text = elem.get("text", "")
            elem_desc = elem.get("content_desc", "")

            user_content += f"Element {i+1}. {elem_id}:\n"
            user_content += f"- Type: {elem_type}\n"

            if elem_text:
                user_content += f"- Text: \"{elem_text}\"\n"

            if elem_desc:
                user_content += f"- Description: \"{elem_desc}\"\n"

            user_content += f"- Clickable: {elem.get('clickable', False)}\n"
            user_content += f"- Enabled: {elem.get('enabled', True)}\n"

            bounds = elem.get("bounds", [])
            if len(bounds) >= 4:
                user_content += f"- Position: {bounds}\n"

            user_content += "\n"

        # Add instructions for the response format
        user_content += """Analyze each element's relevance to the query.
        Respond with a JSON object containing:
        1. "detected_action": The primary action the user wants to perform (click, input, select, view, etc.)
        2. "rankings": An array of objects with:
           - "element_index": Index of the element (starting from 0)
           - "element_id": ID of the element
           - "confidence": A score between 0-1 indicating relevance
           - "reasoning": Brief explanation of why this element matches or doesn't match
           - "action_type": The specific action to perform on this element (click, input, select, view, etc.)

        Format your response as valid JSON only, with no other text."""

        # Create message for the API
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_content
                }
            ]
        }

        messages = [system_message, user_message]

        # Query the LLM
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.1  # Low temperature for consistent results
            )

            content = completion.choices[0].message.content

            # Extract JSON from the response
            json_data = self._extract_json(content)

            # Extract detected action
            detected_action = json_data.get("detected_action", "click")

            # Process the rankings
            if "rankings" in json_data:
                processed_elements = []

                for rank in json_data["rankings"]:
                    element_index = rank.get("element_index", 0)

                    # Ensure index is valid
                    if 0 <= element_index < len(elements):
                        # Copy the element and add LLM analysis
                        element = elements[element_index].copy()
                        element["llm_confidence"] = rank.get("confidence", 0.0)
                        element["reasoning"] = rank.get("reasoning", "")
                        element["element_id"] = rank.get(
                            "element_id", element.get("id", ""))

                        # Add action type information
                        element["action_type"] = rank.get(
                            "action_type", detected_action)

                        processed_elements.append(element)

                # Cache the result
                self.cache[cache_key] = processed_elements
                return processed_elements

            # Fallback: return elements with default confidence
            return elements

        except Exception as e:
            print(f"Error analyzing elements with LLM: {e}")
            # Return original elements with default confidence
            return [
                {**elem, "llm_confidence": 0.5, "reasoning": "LLM analysis failed"}
                for elem in elements
            ]

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON data from LLM response text"""
        try:
            # Try to find JSON block
            import re
            json_match = re.search(r'```json\n([\s\S]*?)\n```', text)

            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON directly
                json_match = re.search(r'(\{[\s\S]*\})', text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text

            # Parse JSON
            return json.loads(json_str)

        except Exception as e:
            print(f"Error extracting JSON: {e}")
            print(f"Original text: {text}")
            return {"rankings": []}


class SimpleRLModel(nn.Module):
    """Simple RL model for adjusting confidence scores"""

    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # Middle layer with 8 units
            nn.ReLU(),         # Activation function
            nn.Linear(8, 1)    # Output layer
        )

    def forward(self, x):
        return self.net(x)


class UINavigator:
    """Main class for LLM-based UI navigation"""

    def __init__(self,
                 vector_store: Optional[VectorStore] = None,
                 embedder: Optional[ElementEmbedder] = None,
                 llm: Optional[MistralHFConnector] = None,
                 config: Optional[Dict] = None,
                 rl_model_path: str = "model/rl_model.pt"):
        """Initialize the Navigator with its components"""
        self.config = config or {}

        # Initialize components
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or ElementEmbedder()
        self.llm = llm or MistralHFConnector()

        # Navigation history
        self.history = {}

        # Initialize RL model if available
        self.rl_model = None
        if os.path.exists(rl_model_path):
            try:
                self.rl_model = SimpleRLModel()

                # Get device information
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")

                # Load with appropriate device mapping
                self.rl_model.load_state_dict(torch.load(
                    rl_model_path, map_location=self.device))

                # Move model to device
                self.rl_model = self.rl_model.to(self.device)
                self.rl_model.eval()

                print(f"Loaded RL model from {rl_model_path} to {self.device}")
            except Exception as e:
                print(f"Error loading RL model: {e}")
                self.rl_model = None
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "avg_processing_time": 0,
            "vector_precision": 0,
            "llm_precision": 0
        }

        # Load navigation history if available
        self._load_history()

    def process_query(self, query: str, screen_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a navigation query and find matching UI elements"""
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
            ranked_elements = self._analyze_with_llm(
                query, candidates_with_actions, screen_metadata)

            # 4. Calculate final confidence scores
            actions = self._calculate_confidence_scores(
                ranked_elements, query, screen_metadata)

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

    def record_feedback(self, element_id: str, screen_id: str, success: bool, action_type: Optional[str] = None) -> bool:
        """Record user feedback for reinforcement learning"""
        try:
            # Create a history key
            key = f"{screen_id}_{element_id}"

            # Initialize if not exists
            if key not in self.history:
                self.history[key] = {
                    "interactions": 0,
                    "successes": 0,
                    "actions": {},
                    "queries": [],
                }

            # Update counts
            self.history[key]["interactions"] += 1
            if success:
                self.history[key]["successes"] += 1

            # Update action statistics if provided
            if action_type:
                if "actions" not in self.history[key]:
                    self.history[key]["actions"] = {}

                if action_type not in self.history[key]["actions"]:
                    self.history[key]["actions"][action_type] = {
                        "count": 0,
                        "success_count": 0
                    }

                self.history[key]["actions"][action_type]["count"] += 1
                if success:
                    self.history[key]["actions"][action_type]["success_count"] += 1

            # Update vector store for future retrievals
            self.vector_store.update_feedback(element_id, success)

            # Save history periodically
            if self.history[key]["interactions"] % 5 == 0:
                self._save_history()

            return True

        except Exception as e:
            print(f"Error recording feedback: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics

    def _add_action_prediction(self, query: str, elements: List[Dict]) -> List[Dict]:
        """Add action type predictions to elements"""
        try:
            # Check if embedder has action prediction methods
            if hasattr(self.embedder, 'predict_actions_for_elements'):
                return self.embedder.predict_actions_for_elements(query, elements)
            elif hasattr(self.embedder, 'extract_action_from_query'):
                # Fall back to simple prediction
                action_type = self.embedder.extract_action_from_query(query)
                input_value = None

                # Try to extract input value if it's an input action
                if action_type == "input" and hasattr(self.embedder, 'extract_input_value'):
                    input_value = self.embedder.extract_input_value(query)

                # Add action type to all elements
                for elem in elements:
                    elem["action_type"] = action_type
                    elem["action_confidence"] = 0.8

                    # Add input value if available
                    if action_type == "input" and input_value:
                        elem["action_parameters"] = {"value": input_value}

                return elements
            else:
                # Fallback if embedder doesn't have action prediction
                return self._extract_action_manually(query, elements)
        except Exception as e:
            print(f"Error predicting actions: {e}")
            return elements

    def _extract_base_action_type(self, query: str) -> str:
        """Extract the base action type from a query"""
        try:
            # Use embedder if it has the method
            if hasattr(self.embedder, 'extract_action_from_query'):
                return self.embedder.extract_action_from_query(query)
            else:
                # Manual extraction
                import re
                query_lower = query.lower()

                # Define action patterns
                action_patterns = [
                    (r'\b(click|tap|press|select|choose)\b', 'click'),
                    (r'\b(type|enter|input|fill|write)\b', 'input'),
                    (r'\b(scroll|swipe)\b', 'scroll'),
                    (r'\b(view|show|display|see|look at)\b', 'view'),
                    (r'\b(transfer|send|move)\s+(money|funds|balance|cash)\b', 'transfer'),
                    (r'\b(search|find|look for)\b', 'search'),
                    (r'\b(filter|sort)\b', 'filter'),
                    (r'\b(navigate|go to|open|visit)\b', 'navigate'),
                ]

                # Check each pattern
                for pattern, action in action_patterns:
                    if re.search(pattern, query_lower):
                        return action

                # Default to click
                return "click"
        except Exception as e:
            print(f"Error extracting action type: {e}")
            return "click"

    def _extract_action_manually(self, query: str, elements: List[Dict]) -> List[Dict]:
        """Manual fallback for action type extraction"""
        action_type = self._extract_base_action_type(query)

        # Extract input value if relevant
        input_value = None
        if action_type == "input":
            import re
            query_lower = query.lower()

            # Patterns to extract quoted text or text after specific phrases
            patterns = [
                # Match quoted text
                r'(?:type|enter|input|fill|write)\s+[\'"]([^\'"]+)[\'"]',
                # Match unquoted text
                r'(?:type|enter|input|fill|write)\s+(?:the\s+)?(?:text|value|words?)?\s*:?\s*([a-zA-Z0-9\s]+)',
            ]

            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    input_value = match.group(1).strip()
                    break

        # Add action type to all elements
        updated_elements = []
        for elem in elements:
            elem_copy = elem.copy()
            elem_copy["action_type"] = action_type
            elem_copy["action_confidence"] = 0.7

            # Special handling for input elements
            if action_type == "input" and input_value:
                elem_copy["action_parameters"] = {"value": input_value}

                # Boost confidence for actual input elements
                if elem_copy.get("type", "").lower() in ["input", "textarea"]:
                    elem_copy["action_confidence"] = 0.9

            # Special handling for select elements
            elif action_type == "select" and elem_copy.get("type", "").lower() in ["select", "dropdown"]:
                elem_copy["action_confidence"] = 0.9

            updated_elements.append(elem_copy)

        return updated_elements

    def _retrieve_similar_elements(self, query: str, elements: List[Dict]) -> List[Dict]:
        """Retrieve relevant elements using vector similarity"""
        # Generate embeddings for the query
        query_embedding = self.embedder.encode_query(query)

        # Generate embeddings for all elements
        element_embeddings = self.embedder.encode_elements(elements)

        # Add elements to vector store temporarily
        self.vector_store.add_elements(elements, element_embeddings)

        # Search for similar elements
        vector_candidates = self.vector_store.search(
            query_embedding,
            k=min(5, len(elements))  # Get top 5 or fewer
        )

        return vector_candidates

    def _analyze_with_llm(self, query: str, elements: List[Dict],
                          screen_metadata: Dict) -> List[Dict]:
        """Analyze elements with LLM for semantic matching"""
        # Extract minimal screen context
        screen_context = {
            "screen_id": screen_metadata.get("screen_id", "unknown"),
            "path": screen_metadata.get("path", ""),
            "url": screen_metadata.get("url", "")
        }

        # Get LLM analysis
        return self.llm.analyze_elements(query, elements, screen_context)

    def _calculate_confidence_scores(self, elements: List[Dict],
                                     query: str, screen_metadata: Dict) -> List[Dict]:
        """Calculate final confidence scores using multiple signals"""
        actions = []
        screen_id = screen_metadata.get("screen_id", "unknown")

        for elem in elements:
            # Get scores from different sources
            vector_score = elem.get("vector_score", 0.5)
            llm_score = elem.get("llm_confidence", 0.5)
            action_confidence = elem.get("action_confidence", 0.7)

            # Get historical success rate if available
            element_id = elem.get("id", "unknown")
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
            llm_weight = 0.5
            vector_weight = 0.25
            action_weight = 0.15
            history_weight = 0.1

            combined_score = (
                llm_weight * llm_score +
                vector_weight * vector_score +
                action_weight * action_confidence +
                history_weight * history_score
            )

            # Apply RL model adjustment if available
            if self.rl_model is not None:
                try:
                    with torch.no_grad():
                        # Get model device
                        model_device = next(self.rl_model.parameters()).device

                        # Create feature tensor on the correct device
                        features = torch.tensor(
                            [[combined_score]], dtype=torch.float).to(model_device)

                        # Get RL adjustment
                        adjustment = self.rl_model(features)

                        # Move result to CPU for item() call
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
                "reasoning": elem.get("reasoning", "")
            }

            # Add action parameters if available
            if action_params:
                action["action_parameters"] = action_params

            actions.append(action)

        # Sort by confidence
        actions.sort(key=lambda x: x["confidence"], reverse=True)

        return actions

    def _update_metrics(self, processing_time: float) -> None:
        """Update performance metrics"""
        # Update average processing time
        prev_avg = self.metrics["avg_processing_time"]
        prev_count = self.metrics["total_queries"] - 1

        if prev_count > 0:
            new_avg = (prev_avg * prev_count + processing_time) / \
                self.metrics["total_queries"]
            self.metrics["avg_processing_time"] = new_avg
        else:
            self.metrics["avg_processing_time"] = processing_time

    def _save_history(self) -> None:
        """Save interaction history to disk"""
        history_path = os.path.join(os.path.dirname(
            __file__), "..", "data", "navigation_history.json")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(history_path), exist_ok=True)

            # Save to file
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    def _load_history(self) -> None:
        """Load interaction history from disk"""
        history_path = os.path.join(os.path.dirname(
            __file__), "..", "data", "navigation_history.json")

        try:
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            self.history = {}
