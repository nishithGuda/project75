# backend2/core/embed_pipeline.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def embed_training_data(data_path="processed_data/training/llm_training_data.json",
                        out_path="processed_data/training/ui_embeddings.json",
                        model_name="all-MiniLM-L6-v2"):
    """Generate and save embeddings for UI elements in the training data"""
    # Create output directory if needed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load SentenceTransformer model
    model = SentenceTransformer(model_name)

    # Load training data
    with open(data_path, "r") as f:
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

            # Generate embedding
            emb = model.encode(label)

            embeddings.append({
                "screen_id": screen_id,
                "element_id": elem["id"],
                "label": label,
                "embedding": emb.tolist(),
                "query": query
            })

    # Save to JSON
    with open(out_path, "w") as f:
        json.dump(embeddings, f, indent=2)

    print(f"Saved {len(embeddings)} element embeddings to {out_path}")
    return embeddings
