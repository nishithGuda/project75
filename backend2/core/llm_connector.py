# backend2/core/llm_connector.py
import requests
import os
import json
from typing import List, Dict, Any, Optional
import time


class MistralConnector:
    """Connects to Mistral AI API for LLM-based UI element reasoning"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "mistral-small-3.1-24b-instruct"):
        # Get API key from environment or parameter
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key is required. Set MISTRAL_API_KEY environment variable.")

        self.model = model
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.cache = {}  # Simple cache for repeated queries

    def analyze_elements(self, query: str, elements: List[Dict],
                         screen_context: Optional[Dict] = None) -> List[Dict]:
        """Analyze UI elements against a user query using the LLM"""
        cache_key = self._get_cache_key(query, elements)

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Prepare system prompt
        system_prompt = """You are an AI expert in UI navigation, helping find the most relevant UI element for a user's query.
        
        Given a user's natural language query and UI element details:
        1. Analyze the semantic match between the query intent and each element's properties
        2. Consider element type, text content, position, and functionality
        3. Rank elements by relevance, providing a confidence score (0-1) and brief reasoning
        
        Use a step-by-step approach to evaluate each element and determine the best match."""

        # Prepare user prompt with element details
        user_prompt = f"User Query: \"{query}\"\n\n"

        # Add screen context if available
        if screen_context:
            user_prompt += "Current Screen Context:\n"
            user_prompt += f"- Screen ID: {screen_context.get('screen_id', 'unknown')}\n"
            user_prompt += f"- Path: {screen_context.get('path', 'unknown')}\n\n"

        # Add element descriptions
        user_prompt += "Available UI Elements:\n\n"

        for i, elem in enumerate(elements):
            elem_id = elem.get("id", f"element_{i}")
            elem_type = elem.get("type", "unknown")
            elem_text = elem.get("text", "")
            elem_desc = elem.get("content_desc", "")

            user_prompt += f"Element {i+1}. {elem_id}:\n"
            user_prompt += f"- Type: {elem_type}\n"

            if elem_text:
                user_prompt += f"- Text: \"{elem_text}\"\n"

            if elem_desc:
                user_prompt += f"- Description: \"{elem_desc}\"\n"

            user_prompt += f"- Clickable: {elem.get('clickable', False)}\n"
            user_prompt += f"- Enabled: {elem.get('enabled', True)}\n"

            bounds = elem.get("bounds", [])
            if len(bounds) >= 4:
                user_prompt += f"- Position: {bounds}\n"

            user_prompt += "\n"

        # Add instructions for the response format
        user_prompt += """Analyze each element's relevance to the query.
        Respond with a JSON object containing:
        1. "rankings": An array of objects with:
           - "element_index": Index of the element (starting from 0)
           - "element_id": ID of the element
           - "confidence": A score between 0-1 indicating relevance
           - "reasoning": Brief explanation of why this element matches or doesn't match
        
        Focus on finding the most semantically relevant element to fulfill the user's query."""

        # Query the LLM
        try:
            result = self._query_llm(system_prompt, user_prompt)
            content = result.get("choices", [{}])[0].get(
                "message", {}).get("content", "")

            # Extract JSON from the response
            json_data = self._extract_json(content)

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

    def _query_llm(self, system_prompt: str, user_prompt: str) -> Dict:
        """Send a query to the Mistral API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Low temperature for consistent results
            "max_tokens": 1000
        }

        # Implement retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, headers=headers, json=data, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    # Exponential backoff
                    wait_time = (2 ** attempt) + 1
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(
                        f"API error: {response.status_code} - {response.text}")
                    break

            except requests.RequestException as e:
                print(f"Request error: {e}")
                time.sleep(1)

        # If we get here, all retries failed
        return {"choices": [{"message": {"content": "{}"}}]}

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

    def _get_cache_key(self, query: str, elements: List[Dict]) -> str:
        """Generate a cache key for a query and elements"""
        # Simplify elements to just their IDs for the cache key
        element_ids = [elem.get("id", str(i))
                       for i, elem in enumerate(elements)]
        return f"{query}:{','.join(element_ids)}"
