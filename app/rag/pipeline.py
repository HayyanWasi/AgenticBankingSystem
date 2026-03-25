import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List
from pathlib import Path
from app.rag.config import DATA_DIR


class RAG:

    @staticmethod
    def ingest_pdf(data_path: Path = DATA_DIR) -> List[Document]:
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
                page.metadata["category"] = "general"

                # reading metadata
                print(f"File : {page.metadata['source']}")
                print(f"Page : {page.metadata['human_page']} of {page.metadata['total_pages']}")
                print(f"Category: {page.metadata['category']}")
                print("---")

                batch.append(page)
                if len(batch) >= BATCH_SIZE:
                    all_docs.extend(batch)
                    batch.clear()

        if batch:
            all_docs.extend(batch)

        print(f"\nTotal documents loaded: {len(all_docs)}")
        return all_docs


ragPipeline = RAG.ingest_pdf()