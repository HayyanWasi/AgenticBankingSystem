from fastapi import APIRouter, Body
from app.agents.bank_manager.agent import master_graph
from langgraph.types import Command

router = APIRouter()

@router.post("/chat")
async def chat_with_bank(
    message: str = Body(..., embed=True), 
    thread_id: str = Body(..., embed=True)
):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if the session is currently paused in a sub-graph
    snapshot = master_graph.get_state(config)
    
    if snapshot.next:
        # Resume the specific sub-graph (KYC, Loan, or Transfer)
        result = master_graph.invoke(Command(resume=message), config)
    else:
        # Start a fresh supervisor evaluation
        result = master_graph.invoke({"messages": [("user", message)]}, config)

    ai_reply = result["messages"][-1].content
    
    return {
        "reply": ai_reply,
        "is_waiting": bool(master_graph.get_state(config).next)
    }