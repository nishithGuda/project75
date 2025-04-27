from datasets import load_dataset
import os
import json
import time

start_time = time.time()
print("Starting dataset load...")

# === Load valid screen_ids from processed_metadata ===
metadata_path = os.path.join("backend2", "datasets", "processed", "base", "processed_metadata.json")
with open(metadata_path, "r") as f:
    metadata = json.load(f)
valid_screen_ids = set(metadata.keys())
print(f"✅ Loaded {len(valid_screen_ids)} screen_ids from processed_metadata.json")

# === Load RICO-Screen2Words dataset ===
ds = load_dataset("rootsautomation/RICO-Screen2Words", split="train")

# === Begin filtering ===
max_samples = 1000
screen2words_data = []

def extract_elements(node, depth=0, max_depth=5):
    """ Recursively extract UI elements from the semantic tree """
    elements = []
    if not isinstance(node, dict):
        return elements
    element = {
        "id": node.get("resource-id", f"elem_{depth}_{len(elements)}"),
        "text": node.get("text", ""),
        "type": node.get("class", "unknown"),
        "clickable": node.get("clickable", False),
        "bounds": node.get("bounds", []),
        "depth": depth
    }
    elements.append(element)
    for child in node.get("children", []):
        elements.extend(extract_elements(child, depth + 1))
    return elements

# === Iterate through and filter ===
for sample in ds:
    # Filter for Finance category
    if sample.get("category", "").strip().lower() != "finance":
        continue

    # Validate screen_id against metadata
    screen_id = str(sample.get("screenId", "unknown"))
    if screen_id not in valid_screen_ids:
        continue

    # Parse semantic annotation (if JSON structure is valid)
    raw = sample.get("semantic_annotations")
    try:
        annotations = json.loads(raw)
    except Exception:
        continue

    # Extract UI elements
    elements = extract_elements(annotations)
    if not elements:
        continue

    # Use up to 3 captions per screen
    captions = sample.get("captions", [])
    if not captions:
        continue

    for caption in captions[:3]:
        screen2words_data.append({
            "query": caption,
            "screen_id": screen_id,
            "elements": elements,
            "target_idx": 0,  # Use heuristics later if needed
            "split": "train"
        })

    if len(screen2words_data) >= max_samples:
        break

elapsed_time = time.time() - start_time
print(f"✅ Collected {len(screen2words_data)} screen2words samples in {elapsed_time:.2f} seconds.")

# === Save output ===
output_dir = os.path.join("backend2", "datasets")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "screen2words_modeb_samples.json")

with open(output_path, "w") as f:
    json.dump(screen2words_data, f, indent=2)

print(f"✅ Saved to {output_path}")
