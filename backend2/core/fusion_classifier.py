# backend2/core/fusion_classifier.py
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from .navigator import UINavigator
from models.llm_model_bert import LLMQueryElementClassifier
from transformers import BertTokenizer
from difflib import SequenceMatcher


class HybridNavigator(UINavigator):
    """Enhanced navigator that combines BERT, RAG and RL approaches"""

    def __init__(self, *args,
                 bert_model_path="model/llm_bert_model.pt",
                 rl_model_path="model/rl_model.pt",
                 element_dim=33,
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

    def _get_cached_result(self, query, screen_id):
        """Try to get cached results for similar queries"""
        if not hasattr(self, 'query_cache'):
            self.query_cache = {}

        # Check for exact match
        cache_key = f"{screen_id}:{query}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # Check for similar queries (simple implementation)
        query_lower = query.lower()
        for key, value in self.query_cache.items():
            if key.startswith(f"{screen_id}:"):
                cached_query = key.split(':', 1)[1].lower()
                # Simple similarity check - could be improved
                if (query_lower in cached_query or cached_query in query_lower) and \
                        abs(len(query_lower) - len(cached_query)) < 5:
                    return value

        return None

    def _cache_result(self, query, screen_id, result):
        """Cache query result"""
        if not hasattr(self, 'query_cache'):
            self.query_cache = {}

        # Keep cache size reasonable
        if len(self.query_cache) > 50:
            # Remove oldest item (simple implementation)
            self.query_cache.pop(next(iter(self.query_cache)))

        cache_key = f"{screen_id}:{query}"
        self.query_cache[cache_key] = result

    def process_query(self, query: str, screen_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a navigation query using RAG + BERT and LLM approaches"""
        start_time = time.time()
        self.metrics["total_queries"] += 1
        screen_id = screen_metadata.get("screen_id", "unknown")

        # Try cache first
        cached_result = self._get_cached_result(query, screen_id)
        if cached_result:
            # Add a note that this is from cache
            cached_result["from_cache"] = True
            return cached_result

        try:
            # Extract elements from metadata
            elements = screen_metadata.get("elements", [])
            if not elements:
                return {
                    "success": False,
                    "error": "No UI elements found in screen metadata",
                    "actions": []
                }

            # 1. Enhance query understanding with BERT
            intent_data = self.enhance_query_understanding(query)

            # 2. Vector-based retrieval
            vector_candidates = self._retrieve_similar_elements(
                query, elements)

            if not vector_candidates:
                return {
                    "success": False,
                    "error": "No relevant elements found for this query",
                    "actions": []
                }

            # 3. RAG with enhanced element contexts
            rag_candidates = self.enhanced_rag_matching(
                query, vector_candidates, intent_data)

            # 4. Add action type prediction using embedder
            candidates_with_actions = self._add_action_prediction(
                query, rag_candidates)

            # 5. LLM-based analysis
            candidates_with_llm = self._analyze_with_llm(
                query, candidates_with_actions, screen_metadata)

            # 6. Extract dropdown values where appropriate
            enhanced_candidates = self._enhance_dropdown_selections(
                query, candidates_with_llm)

            # 7. Calculate final confidence scores with all signals
            actions = self._calculate_confidence_scores(
                enhanced_candidates, query, screen_metadata)

            # 8. Update performance metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time)

            # 9. Add concise descriptions to actions
            for action in actions:
                action["description"] = self._create_action_description(
                    action, query)

            # Extract base action for the query
            base_action = self._extract_base_action_type(query)

            result = {
                "success": True,
                "actions": actions,
                "query": query,
                "detected_action": base_action,
                "processing_time": round(processing_time, 3),
                "vector_candidates": len(vector_candidates),
                "total_elements": len(elements)
            }

            # Cache the result
            self._cache_result(query, screen_id, result)

            return result

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

    def enhance_query_understanding(self, query: str) -> Dict:
        """Extract intent and entities using BERT"""

        # Encode query for BERT
        encoding = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Extract entities using simple patterns
        entities = self._extract_entities(query)

        # Return intent data
        return {
            "query": query,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entities": entities
        }

    def _extract_entities(self, query: str) -> Dict:
        """Extract entities such as account types and numbers"""

        # Define entity patterns
        account_patterns = [
            # Match "Checking (****1234)" format
            r'(Checking|Savings|Credit Card)\s*\(?\*+(\d+)\)?',
            # Match "from Checking account" format
            r'(?:from|to)\s+(Checking|Savings|Credit Card)(?:\s+account)?',
            # Match account numbers
            r'account\s+(?:number)?\s*\*+(\d+)'
        ]

        entities = {
            "source_accounts": [],
            "destination_accounts": [],
            "amounts": [],
            "dates": []
        }

        # Extract accounts
        import re
        for pattern in account_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                account_type = match.group(1) if len(
                    match.groups()) > 0 else None
                account_number = match.group(2) if len(
                    match.groups()) > 1 else None

                # Determine if source or destination based on context
                context_before = query[max(
                    0, match.start() - 20):match.start()]
                if "from" in context_before.lower():
                    entities["source_accounts"].append({
                        "type": account_type,
                        "number": account_number,
                        "span": (match.start(), match.end()),
                        "confidence": 0.9  # Could be based on match quality
                    })
                elif "to" in context_before.lower():
                    entities["destination_accounts"].append({
                        "type": account_type,
                        "number": account_number,
                        "span": (match.start(), match.end()),
                        "confidence": 0.9
                    })
                else:
                    # If no direction cue, check rest of query
                    rest_of_query = query[match.end():].lower()
                    if "to" in rest_of_query:
                        entities["source_accounts"].append({
                            "type": account_type,
                            "number": account_number,
                            "span": (match.start(), match.end()),
                            "confidence": 0.8
                        })
                    elif "from" in rest_of_query:
                        entities["destination_accounts"].append({
                            "type": account_type,
                            "number": account_number,
                            "span": (match.start(), match.end()),
                            "confidence": 0.8
                        })
                    else:
                        # No clear indicators, add to both with lower confidence
                        entities["source_accounts"].append({
                            "type": account_type,
                            "number": account_number,
                            "span": (match.start(), match.end()),
                            "confidence": 0.5
                        })
                        entities["destination_accounts"].append({
                            "type": account_type,
                            "number": account_number,
                            "span": (match.start(), match.end()),
                            "confidence": 0.4
                        })

        return entities

    def enhanced_rag_matching(self, query: str, elements: List[Dict], intent_data: Dict) -> List[Dict]:
        """Use RAG with enhanced context for better element matching"""

        enhanced_elements = []

        for elem in elements:
            # Create enhanced context for the element
            context = self._create_element_context(elem, query, intent_data)

            # Add context to element
            elem_copy = elem.copy()
            elem_copy["enhanced_context"] = context
            elem_copy["rag_score"] = self._calculate_context_match_score(
                context, query, intent_data)

            enhanced_elements.append(elem_copy)

        # Sort by RAG score
        enhanced_elements.sort(key=lambda x: x.get(
            "rag_score", 0), reverse=True)

        return enhanced_elements

    def _create_element_context(self, elem: Dict, query: str, intent_data: Dict) -> str:
        """Create enhanced context for an element based on query and intent"""

        # Get element properties
        elem_type = elem.get("type", "").lower()
        elem_text = elem.get("text", "")
        elem_id = elem.get("id", "")

        # Create context based on element type
        context = ""

        if elem_type in ["select", "dropdown"]:
            context += "This element allows selecting from multiple options. "

            # Check for from/to patterns
            if "from" in elem_id.lower():
                context += "This appears to be a source selection dropdown. "

                # Add source account info if available
                source_accounts = intent_data.get(
                    "entities", {}).get("source_accounts", [])
                if source_accounts:
                    sa = source_accounts[0]
                    if sa.get("type") and sa.get("number"):
                        context += f"The query mentions {sa['type']} (****{sa['number']}) as a source account. "
                    elif sa.get("type"):
                        context += f"The query mentions {sa['type']} as a source account. "

            elif "to" in elem_id.lower():
                context += "This appears to be a destination selection dropdown. "

                # Add destination account info if available
                dest_accounts = intent_data.get(
                    "entities", {}).get("destination_accounts", [])
                if dest_accounts:
                    da = dest_accounts[0]
                    if da.get("type") and da.get("number"):
                        context += f"The query mentions {da['type']} (****{da['number']}) as a destination account. "
                    elif da.get("type"):
                        context += f"The query mentions {da['type']} as a destination account. "

            # Add options information
            if elem_text:
                context += f"The available options include: {elem_text}. "

        # For other element types
        elif elem_type in ["input", "textbox"]:
            context += "This element allows entering text. "
        elif elem_type in ["button"]:
            context += "This element performs an action when clicked. "

        return context

    def _calculate_context_match_score(self, context: str, query: str, intent_data: Dict) -> float:
        """Calculate how well the context matches the query intent"""

        # Simple implementation using string similarity
        similarity = SequenceMatcher(
            None, context.lower(), query.lower()).ratio()

        # Check for entity matches in context
        entities = intent_data.get("entities", {})
        entity_bonus = 0.0

        # Check for source account mentions
        for sa in entities.get("source_accounts", []):
            if sa.get("type") and sa.get("type").lower() in context.lower():
                entity_bonus += 0.15
            if sa.get("number") and sa.get("number") in context.lower():
                entity_bonus += 0.15

        # Check for destination account mentions
        for da in entities.get("destination_accounts", []):
            if da.get("type") and da.get("type").lower() in context.lower():
                entity_bonus += 0.15
            if da.get("number") and da.get("number") in context.lower():
                entity_bonus += 0.15

        # Combine similarity and entity bonus
        return similarity * 0.7 + entity_bonus

    def _enhance_dropdown_selections(self, query: str, elements: List[Dict]) -> List[Dict]:
        """Extract dropdown values for select elements"""

        enhanced_elements = []

        for elem in elements:
            elem_copy = elem.copy()

            # Only process dropdown/select elements
            if elem.get("type", "").lower() in ["select", "dropdown"]:
                # Extract options from the element text
                options = self._extract_dropdown_options(elem)

                # If options were extracted, find the best match
                if options:
                    value = self._match_dropdown_option(query, options, elem)

                    if value:
                        # Add as action parameter
                        elem_copy["action_parameters"] = {"value": value}
                        # Higher confidence since we found a match
                        elem_copy["action_confidence"] = 0.8

            enhanced_elements.append(elem_copy)

        return enhanced_elements

    def _extract_dropdown_options(self, elem: Dict) -> List[str]:
        """Extract options from dropdown element text"""

        elem_text = elem.get("text", "")
        options = []

        # Try different delimiters
        if ")" in elem_text:
            # Format like "Option1 (****1234), Option2 (****5678)"
            parts = elem_text.split(")")
            for part in parts:
                part = part.strip()
                if part:
                    # Add the closing parenthesis back
                    options.append(part + ")")
        elif "," in elem_text:
            # Simple comma-separated list
            options = [opt.strip()
                       for opt in elem_text.split(",") if opt.strip()]
        else:
            # Try space as delimiter
            options = [opt.strip() for opt in elem_text.split() if opt.strip()]

        return options

    def _match_dropdown_option(self, query: str, options: List[str], elem: Dict) -> Optional[str]:
        """Find the best matching option for the query"""

        query_lower = query.lower()
        matches = []

        # Common account types to search for
        account_types = ["checking", "savings", "credit card"]
        account_numbers = []

        # Extract account numbers from query using regex
        import re
        number_pattern = r'\(\*+(\d+)\)'
        for match in re.finditer(number_pattern, query_lower):
            account_numbers.append(match.group(1))

        # Score each option
        for option in options:
            option_lower = option.lower()
            score = 0.0

            # Check for account type matches
            for acc_type in account_types:
                if acc_type in query_lower and acc_type in option_lower:
                    score += 0.5

            # Check for account number matches
            for number in account_numbers:
                if number in option_lower:
                    score += 0.7

            # Check for "from" and "to" context
            elem_id = elem.get("id", "").lower()
            if "from" in elem_id and "from" in query_lower:
                score += 0.3
            elif "to" in elem_id and "to" in query_lower:
                score += 0.3

            # Add the option and score to matches
            matches.append((option, score))

        # Sort by score and get the best match
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return the best match if score is significant
        if matches and matches[0][1] > 0.3:
            return matches[0][0]

        return None

    def _calculate_confidence_scores(self, elements: List[Dict],
                                     query: str, screen_metadata: Dict) -> List[Dict]:
        """Calculate final confidence scores balancing accuracy and diversity"""
        actions = []
        screen_id = screen_metadata.get("screen_id", "unknown")

        # Track recently shown elements for diversity
        recently_shown_elements = self._get_recently_shown_elements(screen_id)

        # Get BERT predictions for all elements
        bert_scores = self._get_bert_scores(query, elements)

        for elem in elements:
            # Get scores from different sources
            vector_score = elem.get("vector_score", 0.5)
            llm_score = elem.get("llm_confidence", 0.5)
            rag_score = elem.get("rag_score", 0.5)
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

            # Calculate diversity bonus
            diversity_bonus = 0.0
            if element_id not in recently_shown_elements:
                diversity_bonus = 0.05  # Smaller diversity bonus

            # STRONGLY weight LLM scores based on evaluation results
            llm_weight = 0.70      # Increased from 0.35
            bert_weight = 0.10      # Reduced from 0.35
            vector_weight = 0.10     # Reduced from 0.15
            rag_weight = 0.05        # New component
            history_weight = 0.025    # Reduced from 0.15
            diversity_weight = 0.025  # Smaller diversity weight

            # Calculate accuracy-focused score
            combined_score = (
                llm_weight * llm_score +
                bert_weight * bert_score +
                vector_weight * vector_score +
                rag_weight * rag_score +
                history_weight * history_score +
                diversity_weight * diversity_bonus
            )

            # Create action object
            action = {
                "id": element_id,
                "type": elem.get("type", "unknown"),
                "text": elem.get("text", ""),
                "confidence": round(float(combined_score), 4),
                "action_type": elem.get("action_type", "click"),
                "reasoning": elem.get("reasoning", ""),
                # Include individual scores for transparency
                "llm_score": round(float(llm_score), 4),
                "vector_score": round(float(vector_score), 4),
                "bert_score": round(float(bert_score), 4),
                "rag_score": round(float(rag_score), 4) if "rag_score" in elem else 0.5,
                "history_score": round(float(history_score), 4),
                "diversity_bonus": round(float(diversity_bonus), 4),
                "raw_text": elem.get("text", "")  # Store for similarity check
            }

            # Add action parameters if present
            if elem.get("action_parameters"):
                action["action_parameters"] = elem.get("action_parameters")

            actions.append(action)

        # Sort by confidence
        actions.sort(key=lambda x: x["confidence"], reverse=True)

        # Apply a second diversity pass to ensure we don't return very similar items
        # This ensures we still get diversity without sacrificing accuracy too much
        diverse_actions = []
        for action in actions:
            # Skip if too similar to an already selected action
            if not self._is_too_similar_to_selected(action, diverse_actions):
                diverse_actions.append(action)

            # Stop once we have 3 diverse actions
            if len(diverse_actions) >= 3:
                break

        # If we don't have enough diverse actions, add more from the original list
        while len(diverse_actions) < min(3, len(actions)):
            for action in actions:
                if action not in diverse_actions:
                    diverse_actions.append(action)
                    break

        # Track selected elements for future diversity
        if diverse_actions and len(diverse_actions) > 0:
            self._update_recent_recommendations(
                screen_id, diverse_actions[0]["id"])

        return diverse_actions

    def _is_too_similar_to_selected(self, candidate, selected_actions, threshold=0.8):
        """Check if a candidate action is too similar to already selected actions"""
        if not selected_actions:
            return False

        candidate_text = candidate.get("raw_text", "").lower()
        if not candidate_text:
            return False

        for selected in selected_actions:
            selected_text = selected.get("raw_text", "").lower()
            if not selected_text:
                continue

            # Check text similarity using a simple ratio
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(
                None, candidate_text, selected_text).ratio()

            if similarity > threshold:
                return True

        return False

    def _get_bert_scores(self, query, elements):
        """Get confidence scores from BERT model with batching"""
        scores = {}

        # Skip if no elements
        if not elements:
            return scores

        # Process in a single batch instead of one by one
        element_features = []
        element_ids = []

        for element in elements:
            # Extract features
            features = self._extract_bert_features(element)
            element_features.append(features)
            element_ids.append(element.get("id", "unknown"))

        # Convert to tensors
        elements_tensor = torch.tensor(
            element_features, dtype=torch.float).to(self.device)

        # Tokenize query once
        encoding = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get predictions for all elements at once
        with torch.no_grad():
            try:
                # Process in smaller batches if needed
                batch_size = 8
                for i in range(0, len(elements_tensor), batch_size):
                    batch_features = elements_tensor[i:i+batch_size]
                    # Repeat query for each element in batch
                    batch_input_ids = input_ids.repeat(len(batch_features), 1)
                    batch_attention_mask = attention_mask.repeat(
                        len(batch_features), 1)

                    logits = self.bert_model(
                        query_input_ids=batch_input_ids,
                        query_attention_mask=batch_attention_mask,
                        element_features=batch_features
                    )

                    # Convert to probabilities
                    probs = torch.sigmoid(logits).cpu().tolist()

                    # Store scores
                    for j, prob in enumerate(probs):
                        element_id = element_ids[i+j]
                        scores[element_id] = prob

            except Exception as e:
                print(f"Error in batch BERT prediction: {e}")
                # Fall back to individual processing
                for element in elements:
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

    def _get_recently_shown_elements(self, screen_id):
        """Track recently shown elements to promote diversity"""
        if not hasattr(self, 'recent_recommendations'):
            self.recent_recommendations = {}

        # Get recently shown elements for this screen
        recent = self.recent_recommendations.get(screen_id, [])

        # Only keep the 3 most recent recommendations
        if len(recent) > 3:
            recent = recent[-3:]

        self.recent_recommendations[screen_id] = recent
        return set(recent)

    def _update_recent_recommendations(self, screen_id, element_id):
        """Update tracking of recently shown recommendations"""
        if not hasattr(self, 'recent_recommendations'):
            self.recent_recommendations = {}

        if screen_id not in self.recent_recommendations:
            self.recent_recommendations[screen_id] = []

        self.recent_recommendations[screen_id].append(element_id)

        # Keep only the last 3
        self.recent_recommendations[screen_id] = self.recent_recommendations[screen_id][-3:]

    def _create_action_description(self, action: Dict, query: str) -> str:
        """Create a concise, informative action description"""

        action_type = action.get("action_type", "click")
        action_text = action.get("text", "")

        # For dropdown/select elements
        if action_type == "select" and action.get("action_parameters"):
            value = action["action_parameters"].get("value", "")
            if value:
                # Truncate value if too long
                if len(value) > 30:
                    value = value[:27] + "..."
                return f"Select {value}"

        # Create default descriptions
        if action_type == "select":
            # Extract first option if available
            options = self._extract_dropdown_options(action)
            if options and len(options) > 0:
                first_option = options[0]
                # Truncate if too long
                if len(first_option) > 20:
                    first_option = first_option[:17] + "..."

                return f"Select {first_option}" + ("..." if len(options) > 1 else "")
            else:
                return f"Select from dropdown"

        elif action_type == "click":
            text = action_text[:30] + ("..." if len(action_text) > 30 else "")
            return f"Click {text or action.get('id', 'button')}"

        elif action_type == "input":
            text = action_text[:30] + ("..." if len(action_text) > 30 else "")
            return f"Enter text in {text or 'input field'}"

        # Default
        text = action_text[:30] + ("..." if len(action_text) > 30 else "")
        return f"{action_type.capitalize()} {text or action.get('id', 'element')}"
