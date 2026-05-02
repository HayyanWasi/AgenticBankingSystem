import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            collection_name="banking_docs",
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db"  # ✅ persists to disk
        )      



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

                page.metadata["source"] = pdf_file.name
                page.metadata["human_page"] = page.metadata["page"] + 1
                # page.metadata["category"] = "general"

                # reading metadata

                print(f"File : {page.metadata['source']}")
                print(f"Page : {page.metadata['human_page']} of {page.metadata['total_pages']}")
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
            chunk_size=CHUNK_SIZE,  
            chunk_overlap=CHUNK_OVERLAP,  
            add_start_index=True,  
        )

        all_splits = text_splitter.split_documents(all_docs)
        print(f"Split blog post into {len(all_splits)} sub-documents.")
        print("Starting ingestion into ChromaDB...")
        batch_size = 100
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i : i + batch_size]
            self.vector_store.add_documents(documents=batch)
            print(f"Progress: {i + len(batch)} / {len(all_splits)} chunks ingested.")
            print("Ingestion complete.")
            return [f"Ingested {len(all_splits)} chunks"]


        
        # document_ids = self.vector_store.add_documents(documents=all_splits)  
        # return document_ids



if __name__ == "__main__":
    rag = RAG()
    rag.ingest_pdf()