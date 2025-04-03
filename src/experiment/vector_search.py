from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import json
import pathlib
from dataclasses import dataclass, asdict
import pickle


@dataclass
class CachedEmbedding:
    """Data class for storing cached embeddings."""

    task_id: int
    embedding: np.ndarray
    task_text: str


class VectorSearch:
    """Class for handling vector search-based few-shot example selection."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[pathlib.Path] = None,
        training_examples: Optional[List[Dict]] = None,
    ):
        """Initialize the vector search with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger("VectorSearch")
        self.cache_dir = cache_dir or pathlib.Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = (
            self.cache_dir / f"{model_name.replace(':', '_')}_embeddings.pkl"
        )
        self.embeddings_cache: Dict[int, CachedEmbedding] = {}
        self._load_cache()
        
        # Precompute embeddings if training examples are provided
        if training_examples:
            self.precompute_embeddings(training_examples)

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure logging for the experiment."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("VectorSearch")

    def _load_cache(self) -> None:
        """Load embeddings from cache file if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    # Convert loaded data back to CachedEmbedding objects
                    self.embeddings_cache = {
                        task_id: CachedEmbedding(**data)
                        for task_id, data in cached_data.items()
                    }
                self.logger.info(
                    f"Loaded {len(self.embeddings_cache)} embeddings from cache"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self) -> None:
        """Save embeddings to cache file."""
        try:
            # Convert CachedEmbedding objects to dictionaries for serialization
            cache_data = {
                task_id: asdict(cached_embedding)
                for task_id, cached_embedding in self.embeddings_cache.items()
            }
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            self.logger.info(f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a given text."""
        return self.model.encode(text)

    def get_task_embedding(self, task: Dict) -> np.ndarray:
        """Get embedding for a task by combining prompt and test cases."""
        task_id = task["task_id"]
        task_text = f"{task['prompt']}"

        # Check if embedding is in cache
        if task_id in self.embeddings_cache:
            cached = self.embeddings_cache[task_id]
            if cached.task_text == task_text:
                return cached.embedding
            else:
                self.logger.warning(
                    f"Task text mismatch for task_id {task_id}, recomputing embedding"
                )

        # Compute new embedding
        embedding = self.get_embedding(task_text)

        # Cache the embedding
        self.embeddings_cache[task_id] = CachedEmbedding(
            task_id=task_id, embedding=embedding, task_text=task_text
        )

        # Save cache after adding new embedding
        self._save_cache()

        return embedding

    def precompute_embeddings(self, tasks: List[Dict]) -> None:
        """Precompute and cache embeddings for a list of tasks."""
        self.logger.info(f"Precomputing embeddings for {len(tasks)} tasks")
        for task in tasks:
            self.get_task_embedding(task)
        self.logger.info("Finished precomputing embeddings")

    def find_similar_examples(
        self,
        target_task: Dict,
        candidate_examples: List[Dict],
        num_examples: int,
    ) -> List[Dict]:
        """Find similar examples using cosine similarity."""
        if not candidate_examples:
            self.logger.warning("No candidate examples provided")
            return []

        # Get embedding for target task
        target_embedding = self.get_task_embedding(target_task)

        # Get embeddings for all candidate examples
        candidate_embeddings = [
            self.get_task_embedding(ex) for ex in candidate_examples
        ]

        # Calculate cosine similarities
        similarities = [
            np.dot(target_embedding, ex_embedding)
            / (np.linalg.norm(target_embedding) * np.linalg.norm(ex_embedding))
            for ex_embedding in candidate_embeddings
        ]

        # Get indices of top k most similar examples
        top_k_indices = np.argsort(similarities)[-num_examples:][::-1]

        # Return the most similar examples
        return [candidate_examples[i] for i in top_k_indices]
