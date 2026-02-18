from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, DirectoryLoader
from typing import List
from pathlib import Path

# load pdf documents from data folder
# directory_path="../data" 
# def load_pdf_documents(directory_path: str)-> List:
#     dir_loader = DirectoryLoader(
#         directory_path,
#         glob="**/*.pdf",
#         loader_cls=PyMuPDFLoader,
#         show_progress=False
#     )

#     pdf_documents = dir_loader.load()
#     return pdf_documents

# pdf_documents = load_pdf_documents(directory_path)
# print(pdf_documents)

# read the pdf documents
def process_all_pdfs(pdf_directory):
    """Process all the PDF in the directory"""
    all_documents=[]
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            #Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents) 
            print(f"loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error: {e}")

    print("Total Documents:", {len(all_documents)})
    return all_documents
BASE_DIR = Path(__file__).resolve().parent.parent  # points to app/
all_pdf_documents = process_all_pdfs(BASE_DIR / "data")