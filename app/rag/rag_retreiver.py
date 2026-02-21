from .vector_store import VectorStore
from .embedding import embedding_manager
from typing import List, Dict, Any


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager):
        """
        Initialize the retriver


        Args:
        vectorStore: Vector store containing document embeddings
        embeddings_manegr: Manager for generating queries
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.4)-> List[Dict[str, Any]]:
        """
        Retrieve documents for a query

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            int: Number of documents to retrieve
            score_threshold: Minimum similarity threshold record
        Return:
            List of dictionaries containing retreived documents and metadata
        """

        print(f"Retrieving documents for query: {query}")
        print(f"Top k: {top_k}, Score threshold :{score_threshold} ")

        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        #search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = top_k
            )
    
            retrieved_docs =[]

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                scores = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, scores)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id':document,
                            'content':metadata,
                            'similarity_score':similarity_score,
                            'distance':distance,
                            'rank':i + 1 
                        })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

            else:
                print("No documents found")

            return retrieved_docs
               
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
                
vector_store = VectorStore()
rag_retriever = RAGRetriever(vector_store, embedding_manager)
print(rag_retriever)


result = rag_retriever.retrieve("What are the Limitations on Disclosure of Account Numbers?", top_k=5, score_threshold=0.4)

print(result)