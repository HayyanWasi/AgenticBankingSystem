from app.agents.privacy_policy_agent.agent import vector_db
from langchain_core.messages import AIMessage
from app.schemas.bank_manager import SupervisorState
from app.config.privacy_policy_agent_config import privacy_policy_llm as llm

def privacy_policy_node(state: SupervisorState):
    # 1. Get the last user message
    query = state["messages"][-1].content

    # 2. If vector DB is unavailable (PDF missing), respond gracefully
    if vector_db is None:
        return {"messages": [AIMessage(content="The privacy policy document is currently unavailable. Please contact support.")]}

    # 3. Manual RAG retrieval
    retrieved_docs = vector_db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 4. Generate response
    system_prompt = f"Use this context to answer: {context}. Only use the provided data."
    response = llm.invoke([
        ("system", system_prompt),
        ("user", query)
    ])
    
    return {"messages": [AIMessage(content=response.content)]}