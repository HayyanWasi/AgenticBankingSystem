from .vector_store import VectorStore
from .embedding import embedding_manager
from .splitter import chunks
from .loader import all_pdf_documents

vector_store = VectorStore()
print(vector_store)

text = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embedding(text)

vector_store.add_documents(chunks, embeddings)