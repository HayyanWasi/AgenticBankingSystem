import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loader import all_pdf_documents

def split_documents(documents, chunk_size=600, chunk_overlap=100):
    """splitting documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n", " ",""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")


    #show example of chunks
    if split_docs:
        print(f"Content {split_docs[0].page_content[:200]}...")
        print(f"Metadata {split_docs[0].metadata}...")

    return split_docs


chunks= split_documents(all_pdf_documents)
print(chunks)