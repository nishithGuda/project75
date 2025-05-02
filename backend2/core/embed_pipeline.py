# backend2/core/embed_pipeline.py
import os
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


def extract_action_from_query(query: str) -> str:
    """Extract action type from query using regex patterns"""
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
    query_lower = query.lower()
    for pattern, action in action_patterns:
        if re.search(pattern, query_lower):
            return action

    # Default to click if no pattern matches
    return "click"


def extract_input_value(query: str) -> Optional[str]:
    """Extract input value from query for input actions"""
    # Only extract for input actions
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


def create_action_embeddings(model) -> Dict[str, np.ndarray]:
    """Create embeddings for each action type"""
    # Define action types and their descriptions
    action_types = {
        "click": "Click or tap on an element",
        "input": "Type, enter or fill in text in a field",
        "select": "Choose an option from a dropdown or list",
        "view": "View or display information",
        "transfer": "Transfer money or funds between accounts",
        "search": "Search for something specific",
        "filter": "Filter or sort information",
        "navigate": "Navigate to a different page or section"
    }

    embeddings = {}

    # Create rich descriptions for each action type
    action_descriptions = {}
    for action, desc in action_types.items():
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
        desc_embeddings = model.encode(descriptions)

        # Average the embeddings for each action type
        embeddings[action] = np.mean(desc_embeddings, axis=0)

    return embeddings


def embed_training_data(data_path="processed_data/training/llm_training_data.json",
                        out_path="processed_data/training/ui_embeddings.json",
                        action_out_path="processed_data/training/action_embeddings.json",
                        processed_out_path="processed_data/training/llm_training_data_with_actions.json",
                        model_name="all-MiniLM-L6-v2"):
    """Generate and save embeddings for UI elements in the training data"""
    # Create output directory if needed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Load training data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Create action embeddings
    action_embeddings = create_action_embeddings(model)

    # Save action embeddings
    with open(action_out_path, "w") as f:
        json.dump({action: emb.tolist()
                  for action, emb in action_embeddings.items()}, f, indent=2)

    print(f"Saved action embeddings to {action_out_path}")

    # Process training data with action types
    action_counts = {}
    for sample in tqdm(data, desc="Processing action types"):
        # Extract action type from query
        query = sample.get("query", "")
        action_type = extract_action_from_query(query)

        # Add action type to sample
        sample["action_type"] = action_type

        # Track action type counts
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

        # Add action parameters for input actions
        if action_type == "input":
            value = extract_input_value(query)
            if value:
                sample["action_parameters"] = {"value": value}

        # Mark target element with action type
        target_idx = sample.get("target_idx", -1)
        elements = sample.get("elements", [])

        if 0 <= target_idx < len(elements):
            elements[target_idx]["action_type"] = action_type
            elements[target_idx]["is_target"] = True

            # Add action parameters to target element
            if action_type == "input" and "action_parameters" in sample:
                elements[target_idx]["action_parameters"] = sample["action_parameters"]

    # Save processed training data
    with open(processed_out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved processed training data to {processed_out_path}")
    print("Action type distribution:")
    total = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action}: {count} ({count/total:.1%})")

    # Generate element embeddings
    element_embeddings = []
    for sample in tqdm(data, desc="Embedding UI elements"):
        query = sample["query"]
        screen_id = sample["screen_id"]
        elements = sample["elements"]
        action_type = sample.get("action_type", "click")

        for elem in elements:
            # Combine text and content description for embedding
            text = elem.get("text", "")
            desc = elem.get("content_desc", "")
            label = f"{text} {desc}".strip()

            # Skip empty elements
            if not label:
                continue

            # Generate embedding
            emb = model.encode(label)

            # Check if this is a target element
            is_target = elem.get("is_target", False)

            # Add element action type if it's a target
            elem_action = elem.get(
                "action_type", action_type if is_target else "click")

            element_embeddings.append({
                "screen_id": screen_id,
                "element_id": elem["id"],
                "label": label,
                "embedding": emb.tolist(),
                "query": query,
                "action_type": elem_action,
                "is_target": is_target
            })

    # Save to JSON
    with open(out_path, "w") as f:
        json.dump(element_embeddings, f, indent=2)

    print(f"Saved {len(element_embeddings)} element embeddings to {out_path}")
    return element_embeddings, action_embeddings


# If run directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings for UI elements")
    parser.add_argument("--data-path", default="processed_data/training/llm_training_data.json",
                        help="Path to training data")
    parser.add_argument("--out-path", default="processed_data/training/ui_embeddings.json",
                        help="Path to save element embeddings")
    parser.add_argument("--action-out-path", default="processed_data/training/action_embeddings.json",
                        help="Path to save action embeddings")
    parser.add_argument("--processed-out-path", default="processed_data/training/llm_training_data_with_actions.json",
                        help="Path to save processed training data")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name")

    args = parser.parse_args()

    embed_training_data(
        data_path=args.data_path,
        out_path=args.out_path,
        action_out_path=args.action_out_path,
        processed_out_path=args.processed_out_path,
        model_name=args.model_name
    )
