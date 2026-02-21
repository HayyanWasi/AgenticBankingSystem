import chromadb
import os
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from typing import List, Dict, Any
import numpy as np
from .splitter import chunks
from .embedding import embedding_manager





class VectorStore:
    """Manages documents embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data"):
        """Initialize the vector store
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the collection
        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()


    def _initialize_store(self):
        """Initialize ChromaDB client and collection """
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient()

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata = {"description": "PDF document for embedding RAG"}
                )
            print(f"Vector store initialize collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise  

    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray ):
        """Add documents and their embeddings to vector store
        Args:
        documents: List of documents with metadata and embeddings
        embeddings: Corresponding embeddings for the documents

        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")

        print(f"Adding {len(documents)} documents to vector store")


        # prepare data for chromadb

        ids = []
        metadatas = []
        documents_text = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            #Genrate unique ids
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embedding_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=documents_text
            )

            print(f"Added {len(documents)} documents to vector store")
            print("Successfully added documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
            

# vector_store = VectorStore()
# print(vector_store)


# ###Convert text into embeddings
# text = [doc.page_content for doc in chunks]

# # print(text)


# embeddings = embedding_manager.generate_embedding(text)

# #storing in database

# vector_store.add_documents(chunks, embeddings)


