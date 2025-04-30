import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load dataset
DATA_PATH = "processed_data/training/llm_training_data.json"
OUT_PATH = "processed_data/training/ui_embeddings.json"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your LLM training data
with open(DATA_PATH, "r") as f:
    data = json.load(f)

embeddings = []

for sample in tqdm(data, desc="Embedding UI elements"):
    query = sample["query"]
    screen_id = sample["screen_id"]
    elements = sample["elements"]

    for elem in elements:
        # Combine text and content description for embedding
        text = elem.get("text", "")
        desc = elem.get("content_desc", "")
        label = f"{text} {desc}".strip()

        # Skip empty elements
        if not label:
            continue

        emb = model.encode(label)

        embeddings.append({
            "screen_id": screen_id,
            "element_id": elem["id"],
            "label": label,
            "embedding": emb.tolist(),
            "query": query
        })

# Save to JSON
with open(OUT_PATH, "w") as f:
    json.dump(embeddings, f, indent=2)

print(f"Saved {len(embeddings)} element embeddings to {OUT_PATH}")


model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(query):
    return model.encode(query).tolist()
