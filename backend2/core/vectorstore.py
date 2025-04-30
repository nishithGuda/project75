# backend2/core/vectorstore.py
import numpy as np
from typing import List, Dict, Any, Optional
import pickle
import os


class VectorStore:
    """Vector database for efficient similarity search of UI elements"""

    def __init__(self, dimension: int = 384, index_path: Optional[str] = None):
        self.dimension = dimension
        self.elements = []
        self.embeddings = []  # Store embeddings directly
        self.metadata = {}  # Store additional metadata for each element

        if index_path and os.path.exists(f"{index_path}.data"):
            self.load(index_path)

    def add_elements(self, elements: List[Dict], embeddings: np.ndarray) -> List[int]:
        """Add elements and their embeddings to the store"""
        if embeddings.shape[0] == 0 or len(elements) == 0:
            return []

        # Store starting index
        start_idx = len(self.elements)

        # Normalize embeddings for cosine similarity
        normalized_embeddings = self._normalize_embeddings(embeddings)

        # Store elements and embeddings
        for i, (elem, embedding) in enumerate(zip(elements, normalized_embeddings)):
            # Store the element and its embedding
            self.elements.append(elem)
            self.embeddings.append(embedding)

            # Extract and store important metadata for this element
            element_id = elem.get("id", f"element_{start_idx + i}")
            self.metadata[start_idx + i] = {
                "id": element_id,
                "type": elem.get("type", "unknown"),
                "text": elem.get("text", ""),
                "success_count": 0,
                "failure_count": 0
            }

        # Return the indices where elements were inserted
        return list(range(start_idx, start_idx + len(elements)))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for elements similar to the query embedding using cosine similarity"""
        if len(self.elements) == 0 or len(self.embeddings) == 0:
            return []

        # Normalize query embedding
        normalized_query = self._normalize_embeddings(
            query_embedding.reshape(1, -1))[0]

        # Compute cosine similarity with all embeddings
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = np.dot(normalized_query, embedding)
            similarities.append((i, similarity))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top k results
        top_k = similarities[:min(k, len(similarities))]

        # Create results with similarity scores
        results = []
        for idx, similarity in top_k:
            if 0 <= idx < len(self.elements):
                # Get the element
                element = self.elements[idx].copy()

                # Add similarity score
                element["vector_score"] = float(similarity)

                # Add metadata if available
                if idx in self.metadata:
                    meta = self.metadata[idx]
                    element["success_rate"] = self._calculate_success_rate(
                        meta)

                results.append(element)

        return results

    def update_feedback(self, element_id: str, success: bool) -> bool:
        """Update feedback for an element based on user interaction"""
        # Find the element index by id
        idx = -1
        for i, meta in self.metadata.items():
            if meta.get("id") == element_id:
                idx = i
                break

        if idx == -1:
            return False

        # Update success/failure counts
        if success:
            self.metadata[idx]["success_count"] += 1
        else:
            self.metadata[idx]["failure_count"] += 1

        return True

    def save(self, path: str) -> bool:
        """Save the store and metadata to disk"""
        try:
            # Save the elements, embeddings, and metadata
            data = {
                "elements": self.elements,
                "embeddings": self.embeddings,
                "metadata": self.metadata,
                "dimension": self.dimension
            }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save data
            with open(f"{path}.data", "wb") as f:
                pickle.dump(data, f)

            return True
        except Exception as e:
            print(f"Error saving vector store: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load the store and metadata from disk"""
        try:
            # Load the data
            if os.path.exists(f"{path}.data"):
                with open(f"{path}.data", "rb") as f:
                    data = pickle.load(f)

                self.elements = data.get("elements", [])
                self.embeddings = data.get("embeddings", [])
                self.metadata = data.get("metadata", {})
                self.dimension = data.get("dimension", self.dimension)

                return True

            return False

        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        # Calculate L2 norm along each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        normalized = embeddings / np.maximum(norms, 1e-10)
        return normalized.astype(np.float32)

    def _calculate_success_rate(self, metadata: Dict) -> float:
        """Calculate success rate from interaction history"""
        success = metadata.get("success_count", 0)
        failure = metadata.get("failure_count", 0)

        # If no interactions, return neutral score
        if success + failure == 0:
            return 0.5

        # Calculate success rate
        return success / (success + failure)
