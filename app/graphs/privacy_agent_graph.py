from app.agents.privacy_policy_agent.pipeline import RAG
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from app.schemas.bank_manager import SupervisorState

rag_engine = None

def get_rag():
    global rag_engine
    if rag_engine is None:
        rag_engine = RAG()
    return rag_engine

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") # Use 2.0 per previous fixes

def privacy_policy_node(state: SupervisorState):
    # 1. Get the last user message
    query = state["messages"][-1].content
    
    # 2. Manual RAG retrieval (bypassing the need for a complex internal agent loop)
    rag = get_rag()
    retrieved_docs = rag.vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 3. Generate response
    system_prompt = f"Use this context to answer: {context}. Only use the provided data."
    response = llm.invoke([
        ("system", system_prompt),
        ("user", query)
    ])
    
    return {"messages": [AIMessage(content=response.content)]}