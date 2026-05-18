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
        # 1. Resume the graph
        result = master_graph.invoke(Command(resume=message), config)
    else:
        # 2. Fresh start
        result = master_graph.invoke({"messages": [("user", message)]}, config)

    # 3. Check if we just hit ANOTHER interrupt (e.g., asked for the next field)
    new_snapshot = master_graph.get_state(config)
    if new_snapshot.tasks and new_snapshot.tasks[0].interrupts:
        # Get the question from the interrupt itself
        ai_reply = new_snapshot.tasks[0].interrupts[0].value
    else:
        # Get the final message from the LLM
        ai_reply = result["messages"][-1].content
        
    is_waiting = bool(new_snapshot.tasks and new_snapshot.tasks[0].interrupts)
    
    return {
        "reply": ai_reply,
        "is_waiting": is_waiting,
        "thread_id": thread_id
    }