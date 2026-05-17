# app/api/v1/privacy.py
from fastapi import APIRouter, Body
from app.agents.privacy_policy_agent.agent import rag_bot # Import your compiled graph
from langchain_core.messages import HumanMessage, SystemMessage

router = APIRouter()

@router.post("/chat")
async def chat_with_policy(message: str = Body(..., embed=True), thread_id: str = Body(..., embed=True)):
    config = {"configurable": {"thread_id": thread_id}}
    
    # We use the same structure as your main() function
    input_state = {
        "messages": [
            SystemMessage(content="Use the pdf note. Answer questions based ONLY on pdf content."),
            HumanMessage(content=message)
        ]
    }
    
    result = rag_bot.invoke(input_state, config=config)
    
    return {
        "reply": result["messages"][-1].content,
        "thread_id": thread_id
    }