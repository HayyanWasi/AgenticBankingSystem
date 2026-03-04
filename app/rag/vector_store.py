"""
ChromaDB Vector Store — handles storage and querying of document embeddings.
"""

import os
import uuid
import chromadb
import numpy as np
from typing import List, Any

from .config import COLLECTION_NAME, PERSIST_DIR


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(self):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "PDF document embeddings for RAG"},
        )
        print(f"Collection ready: {COLLECTION_NAME}")
        print(f"Existing docs: {self.collection.count()}")

    # ── Write ──────────────────────────────────────────

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Store document chunks and their embeddings."""
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

        ids, metas, texts, embs = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")

            meta = dict(doc.metadata)
            meta["doc_index"] = i
            meta["content_length"] = len(doc.page_content)
            metas.append(meta)

            texts.append(doc.page_content)
            embs.append(emb.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embs,
            metadatas=metas,
            documents=texts,
        )
        print(f"Added {len(documents)} docs → total {self.collection.count()}")

    # ── Read ───────────────────────────────────────────

    def query(self, query_embedding: np.ndarray, top_k: int):
        """Return raw ChromaDB query results."""
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
