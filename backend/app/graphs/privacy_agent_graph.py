from app.agents.privacy_policy_agent.agent import get_retriever
from langchain_core.messages import AIMessage
from app.schemas.bank_manager import SupervisorState
from app.config.privacy_policy_agent_config import privacy_policy_llm as llm

def privacy_policy_node(state: SupervisorState):
    # 1. Get the last user message
    query = state["messages"][-1].content

    # 2. Get retriever dynamically
    retriever = get_retriever()
    if retriever is None:
        return {"messages": [AIMessage(content="The privacy policy document is currently unavailable. Please contact support.")]}

    # 3. Manual RAG retrieval (retriever.invoke returns docs directly)
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 4. Generate response
    system_prompt = f"Use this context to answer: {context}. Only use the provided data."
    response = llm.invoke([
        ("system", system_prompt),
        ("user", query)
    ])
    
    return {"messages": [AIMessage(content=response.content)]}