import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self, embedding_path="processed_data/training/ui_embeddings.json"):
        with open(embedding_path, "r") as f:
            self.embeddings = json.load(f)

        # Extract vectors and metadata
        self.vectors = np.array([e["embedding"] for e in self.embeddings])
        self.meta = [
            {
                "screen_id": e["screen_id"],
                "element_id": e["element_id"],
                "label": e["label"],
                "query": e["query"]
            } for e in self.embeddings
        ]

    def search(self, query_embedding, top_k=5):
        """
        Return top_k UI elements most similar to the query embedding.
        """
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in top_indices:
            result = self.meta[idx].copy()
            result["score"] = float(similarities[idx])
            results.append(result)

        return results
