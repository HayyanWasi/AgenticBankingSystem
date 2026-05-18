from fastapi import APIRouter, Body
from app.agents.bank_manager.agent import master_graph
from langgraph.types import Command

router = APIRouter()

@router.post("/review")
async def review_transaction(
    thread_id: str = Body(..., embed=True), 
    decision: str = Body(..., embed=True) # "approve" or "reject"
):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if there is actually something to review
    snapshot = master_graph.get_state(config)
    if not snapshot.next:
        return {"error": "No pending review found for this thread."}

    # Resume the graph with the admin's decision
    # This sends the "approve"/"reject" string to the interrupt() call
    result = master_graph.invoke(Command(resume=decision), config)
    
    return {
        "message": f"Transaction has been {decision}ed.",
        "final_status": result["messages"][-1].content
    }