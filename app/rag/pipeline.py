import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
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

class RAG:

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.vector_store = VectorStore()


    def ingest_pdf(self, data_path: Path = DATA_DIR) -> List[Document]:

        """Document loader, splitter and embedder for PDF files"""
        pdf_files = list(Path(data_path).glob("**/*.pdf"))

        if not pdf_files:
            print("No PDFs found — nothing to ingest.")
            return []

        all_docs: List[Document] = []
        batch: List[Document] = []
        BATCH_SIZE = 50

        for pdf_file in pdf_files:
            loader = PyMuPDFLoader(str(pdf_file), mode="page")

            for page in loader.lazy_load():

                # updating metadata

                # page.metadata["source"] = pdf_file.name
                # page.metadata["human_page"] = page.metadata["page"] + 1
                # page.metadata["category"] = "general"

                # reading metadata

                # print(f"File : {page.metadata['source']}")
                # print(f"Page : {page.metadata['human_page']} of {page.metadata['total_pages']}")
                # print(f"Category: {page.metadata['category']}")
                # print("---")

                batch.append(page)
                if len(batch) >= BATCH_SIZE:
                    all_docs.extend(batch)
                    batch.clear()

        if batch:
            all_docs.extend(batch)

        print(f"\nTotal documents loaded: {len(all_docs)}")
        print(all_docs)


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,  # chunk size (characters)
            chunk_overlap=CHUNK_OVERLAP,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )

        all_splits = text_splitter.split_documents(all_docs)
        print(f"Split blog post into {len(all_splits)} sub-documents.")
        
        texts = [doc.page_content for doc in all_splits]

        embeddings = self.embedding_model.encode(texts)
        
        document_ids = self.vector_store.add_documents(documents=all_splits, embeddings=embeddings)
        return document_ids



if __name__ == "__main__":
    rag = RAG()
    rag.ingest_pdf()