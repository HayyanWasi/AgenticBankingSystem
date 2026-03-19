import os
import sys
# Add the project root to sys.path to resolve absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document 

from typing import List, Dict, Any
from pathlib import Path
from app.rag.config import DATA_DIR



class RAG:

  
    @staticmethod   
    def ingest_pdf(data_path: Path = DATA_DIR)-> List[Document]:
        pdf_files = list(Path(data_path).glob("**/*.pdf"))
        if not pdf_files:
            print("No PDFs found — nothing to ingest.")
            return

        all_docs: List[Document] = []
        BATCH_SIZE = 50

        batch: List[Document] = []

        for pdf_file in pdf_files:
                loader = PyMuPDFLoader(str(pdf_file), mode="page")

                for page in loader.lazy_load():
                    batch.append(page)
                    if len(batch) >= BATCH_SIZE:
                        all_docs.extend(batch)
                        batch = []

        if batch:
            all_docs.extend(batch)

        print(f"Total documents loaded: {len(all_docs)}")
        return all_docs



ragPipeline = RAG.ingest_pdf()
# print(ragPipeline)