from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from app.rag.pipeline import rag_pipeline

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def rag_simple(query: str, top_k: int = 3) -> str:
    """Retrieve context from the vector store and ask the LLM."""
    results = rag_pipeline.query(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "No relevant context found to answer the question"

    prompt = f"""Use the following context to answer the question.

    Context: 
    {context}

    Question:
    {query}

    Answer:

    """

    response = llm.invoke([prompt])
    return response.content


if __name__ == "__main__":
    answer = rag_simple("What are the Limitations on Disclosure of Account Numbers?")
    print(answer)