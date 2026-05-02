from .pipeline import RAG

rag = RAG()
rag.ingest_pdf()
print("ingestion complete")