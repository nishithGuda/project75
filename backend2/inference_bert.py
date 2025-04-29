import torch
from transformers import BertTokenizer, BertModel
from models.llm_model_bert import LLMQueryElementClassifier

import json
import argparse

MODEL_PATH = "model/llm_bert_model.pt"
ELEMENT_DIM = 20

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = LLMQueryElementClassifier(element_feature_dim=ELEMENT_DIM)

# Load the model and move it to the proper device
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def enhanced_element_features(elem, screen_width=1440, screen_height=2560):
    """
    Extract features that match the trained model's expected input
    Returns exactly 20 features
    """
    features = []

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
    aspect_ratio = width / max(height, 0.001)  # Avoid division by zero

    position_features = [center_x, center_y, width, height, area, aspect_ratio]
    features.extend(position_features)  # 6 features

    # 2. Element properties - 4 features
    clickable = 1.0 if elem.get("clickable", False) else 0.0
    enabled = 1.0 if elem.get("enabled", True) else 0.0
    visible = 1.0 if elem.get("visible", True) else 0.0
    depth = min(elem.get("depth", 0) / 10.0, 1.0)  # Normalized depth

    features.extend([clickable, enabled, visible, depth])  # 4 features

    # 3. Text presence - 2 features
    has_text = 1.0 if elem.get("text", "") else 0.0
    has_desc = 1.0 if elem.get("content_desc", "") else 0.0

    features.extend([has_text, has_desc])  # 2 features

    # 4. Element type - 8 features (simplified from your original 12)
    # Simplify to the 8 most common types for a total of 20 features
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

    # Total: 6 + 4 + 2 + 8 = 20 features
    assert len(
        features) == 20, f"Feature vector should have exactly 20 elements, got {len(features)}"

    return features


def predict_best_element(query, elements):
    predictions = []

    for element in elements:
        # Extract enhanced features - make sure to move to the correct device
        element_feat = torch.tensor(
            enhanced_element_features(element), dtype=torch.float).unsqueeze(0).to(device)

        # Prepare the input - make sure to move to the correct device
        encoding = tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        # Make sure these tensors are on the same device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(
                query_input_ids=input_ids,
                query_attention_mask=attention_mask,
                element_features=element_feat
            )
            # Move result to CPU for standard Python processing
            prob = torch.sigmoid(logits).cpu().item()

        predictions.append({
            "element": element,
            "confidence": prob
        })

    # Sort elements by confidence score (descending)
    ranked = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
    return ranked


def load_elements_from_metadata(screen_id):
    # Try to load real elements from your processed metadata
    metadata_path = "datasets/processed_metadata.json"

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check how metadata is structured - it might be a list, not a dict
        if isinstance(metadata, list):
            for item in metadata:
                if item.get("screen_id") == screen_id:
                    return item.get("elements", [])
        elif isinstance(metadata, dict) and screen_id in metadata:
            return metadata[screen_id].get("elements", [])
        else:
            print(f"Screen ID {screen_id} not found in metadata")

    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON in metadata file")
    except Exception as e:
        print(f"Error loading metadata: {e}")

    print("Using fallback dummy elements")
    return [
        {
            "id": "transfer_btn",
            "type": "button",
            "text": "Transfer",
            "content_desc": "Transfer money between accounts",
            "clickable": True,
            "enabled": True,
            "visible": True,
            "bounds": [100, 500, 300, 550],
            "depth": 3
        },
        {
            "id": "savings_tab",
            "type": "button",
            "text": "Savings",
            "content_desc": "View savings accounts",
            "clickable": True,
            "enabled": True,
            "visible": True,
            "bounds": [350, 200, 500, 250],
            "depth": 2
        },
        {
            "id": "settings_icon",
            "type": "image",
            "text": "",
            "content_desc": "Settings",
            "clickable": True,
            "enabled": True,
            "visible": True,
            "bounds": [900, 50, 950, 100],
            "depth": 1
        },
        {
            "id": "payment_app_card",
            "type": "button",
            "text": "Payment Applications",
            "content_desc": "View payment methods and applications",
            "clickable": True,
            "enabled": True,
            "visible": True,
            "bounds": [200, 600, 500, 700],
            "depth": 2
        }
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test BERT model for UI element prediction")
    parser.add_argument("--query", type=str, required=True,
                        help="Natural language query")
    parser.add_argument("--screen-id", type=str, default="",
                        help="Screen ID to use real elements")

    args = parser.parse_args()
    query = args.query
    screen_id = args.screen_id

    elements = load_elements_from_metadata(screen_id)

    print(f"\nDEBUG: Testing query '{query}' against {len(elements)} elements")
    print(f"Using device: {device}")

    try:
        results = predict_best_element(query, elements)

        print("\nTop Predictions:")
        for i, res in enumerate(results[:3]):
            element = res["element"]
            confidence = res["confidence"]
            print(f"Element: {element['id']}, Type: {element['type']}, " +
                  f"Text: '{element['text']}', Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
