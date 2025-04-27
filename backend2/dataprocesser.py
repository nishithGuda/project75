import os
import json
from tqdm import tqdm

# === Load helper ===
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# === Paths ===
data_dir = "backend2/datasets"
meta_path = os.path.join(data_dir, "processed_metadata.json")
sem_path = os.path.join(data_dir, "processed_semantics.json")
refexp_path = os.path.join(data_dir, "refexp_samples.json")
s2w_path = os.path.join(data_dir, "screen2words_modeb_samples.json")
banking_path = os.path.join(data_dir, "banking77_1000_samples.json")

# === Load all data ===
metadata = load_json(meta_path)
semantics = load_json(sem_path)
refexp = load_json(refexp_path)
s2w = load_json(s2w_path)
banking = load_json(banking_path)

# === Output lists ===
llm_data = []
rl_data = []

# === Bounding box IoU helper ===
def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea)

# === Unified element builder ===
def get_elements(screen_id):
    meta = metadata.get(str(screen_id))
    if not meta:
        return []
    return meta["elements"]

# === RefExp integration ===
for sample in tqdm(refexp, desc="Processing RefExp"):
    screen_id = str(sample["image_id"])
    query = sample["prompt"]
    bbox = sample["target_bounding_box"]

    elements = get_elements(screen_id)
    if not elements:
        continue

    target_idx = -1
    for i, el in enumerate(elements):
        if el["bounds"] and iou(el["bounds"], [
            bbox["xmin"] * 1440,
            bbox["ymin"] * 2560,
            bbox["xmax"] * 1440,
            bbox["ymax"] * 2560
        ]) > 0.5:
            target_idx = i
            break

    if target_idx == -1:
        continue

    llm_data.append({
        "query": query,
        "screen_id": screen_id,
        "elements": elements,
        "target_idx": target_idx,
        "split": "train"
    })

    rl_data.append({
        "state": {
            "query": query,
            "screen_id": screen_id,
            "confidence": 1.0
        },
        "action": "click",
        "reward": 1.0,
        "next_state": None
    })

# === Screen2Words ===
for sample in tqdm(s2w, desc="Processing Screen2Words"):
    llm_data.append(sample)
    rl_data.append({
        "state": {
            "query": sample["query"],
            "screen_id": sample["screen_id"],
            "confidence": 0.8
        },
        "action": "click",
        "reward": 1.0,
        "next_state": None
    })

# === Banking77 ===

# === Save outputs ===
out_dir = os.path.join("processed_data", "training")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "llm_training_data.json"), "w") as f:
    json.dump(llm_data, f, indent=2)

with open(os.path.join(out_dir, "rl_training_data.json"), "w") as f:
    json.dump(rl_data, f, indent=2)

print(f"✅ Saved {len(llm_data)} LLM samples and {len(rl_data)} RL samples to {out_dir}")

# === Save Banking77 separately ===
banking_out = os.path.join(out_dir, "banking77_queries.json")
with open(banking_out, "w") as f:
    json.dump(banking, f, indent=2)
print(f"✅ Saved {len(banking)} Banking77 queries to {banking_out}")

