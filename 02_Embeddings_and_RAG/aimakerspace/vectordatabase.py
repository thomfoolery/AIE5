import datetime
import numpy as np
from enum import Enum
from collections import defaultdict
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio

class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Euclidean distance between two vectors."""
    distance = np.linalg.norm(vector_a - vector_b)
    return 1 / (1 + distance)  # Normalize to 0-1 range
    
class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None, distance_metric=DistanceMetric.COSINE):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.distance_metric = distance_metric

    def insert(self, key: str, vector: np.array, metadata: dict = None) -> None:
        """Insert a vector with optional metadata."""
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
    ) -> List[Tuple[str, float, dict]]:
        """Search for similar vectors and return keys, scores, and metadata."""
        distance_measure: Callable = cosine_similarity if self.distance_metric == DistanceMetric.COSINE else euclidean_distance
        scores = [
            (key, distance_measure(query_vector, vector), self.metadata.get(key, {}))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float, dict]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k)
        if return_as_text:
            return [(result[0], result[2]) for result in results]
        return results

    def retrieve_from_key(self, key: str) -> Tuple[np.array, dict]:
        """Retrieve both vector and metadata for a given key."""
        return self.vectors.get(key, None), self.metadata.get(key, {})

    async def abuild_from_list(
        self, 
        list_of_text: List[str], 
        metadata_list: List[dict] = None,
    ) -> "VectorDatabase":
        """Build database from list of texts with optional metadata."""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for idx, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            # Get base metadata
            metadata = metadata_list[idx] if metadata_list else {}
            # Add timestamp
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            self.insert(text, np.array(embedding), metadata)
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
