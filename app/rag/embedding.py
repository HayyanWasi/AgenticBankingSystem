import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sentence_transformers.SentenceTransformer import SentenceTransformer
import numpy as np


class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-V2"):
        """
        Initialize the manager


        Args:
             model_name: Hugging Face model name for sentence embeddings

        """
        self.model_name = model_name
        self.model = None
        self._load_model()


    def _load_model(self):
        """Load the sentence Transformer model"""
        try:
            print(f"load embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # print(f"Model Loaded sucessfully. Embedding Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}") 
            raise

    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for list of texts

        Args:
        texts: List of text string to embed

        Returns:
        numpy array of embeddings with shape (len(text), embeddings_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generate embeddings for {len(texts)} texts.. ")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embedings with shape: {embeddings.shape}")
        return embeddings

    # def get_embedding_dimension(self) -> int:
    #     """Get the embedding with the shape """
    #     if not self.model:
    #         raise ValueError("Model not loaded")
    #     return self.model.get_sentence_embedding_dimension()


embedding_manager = EmbeddingManager()
print(embedding_manager)