"""
RAG Pipeline — single entry-point for ingestion and retrieval.
"""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vector_store import VectorStore
from .config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    TOP_K,
    SCORE_THRESHOLD,
    DATA_DIR,
)


class RAGPipeline:
    """Load → Split → Embed → Store → Retrieve"""

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.vector_store = VectorStore()

    # ─── INGESTION ─────────────────────────────────────

    def ingest_pdfs(self, data_path: Path = DATA_DIR):
        """Load PDFs, split into chunks, embed, and store in vector DB."""
        pdf_files = list(Path(data_path).glob("**/*.pdf"))
        if not pdf_files:
            print("No PDFs found — nothing to ingest.")
            return

        all_docs = []
        print(f"Found {len(pdf_files)} PDFs")

        for pdf_file in pdf_files:
            try:
                docs = PyMuPDFLoader(str(pdf_file)).load()
                for doc in docs:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["file_type"] = "pdf"
                all_docs.extend(docs)
                print(f"  ✓ {pdf_file.name}: {len(docs)} pages")
            except Exception as e:
                print(f"  ✗ {pdf_file.name}: {e}")

        print(f"Loaded {len(all_docs)} total pages")

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=SEPARATORS,
        )
        chunks = splitter.split_documents(all_docs)
        print(f"Split into {len(chunks)} chunks")

        # Embed
        texts = [c.page_content for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Store
        self.vector_store.add_documents(chunks, embeddings)
        print("Ingestion complete ✓")

    # ─── RETRIEVAL ─────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a question."""
        print(f"Retrieving for: {question}")

        query_embedding = self.model.encode([question])[0]
        results = self.vector_store.query(query_embedding, top_k)

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for i, (doc_id, doc, meta, dist) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                score = 1 - dist
                if score >= score_threshold:
                    retrieved.append(
                        {
                            "id": doc_id,
                            "content": doc,
                            "metadata": meta,
                            "similarity_score": score,
                            "rank": i + 1,
                        }
                    )

        print(f"Retrieved {len(retrieved)} docs (threshold={score_threshold})")
        return retrieved


# ── Module-level singleton ─────────────────────────────
rag_pipeline = RAGPipeline()


if __name__ == "__main__":
    # One-time ingestion: python -m app.rag.pipeline
    rag_pipeline.ingest_pdfs()