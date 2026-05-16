from fastapi import APIRouter, Body
from app.graphs.transfer_graph import transfer_workflow
from langgraph.types import Command

router = APIRouter()

@router.post("/initiate")
async def initiate_transfer(
    sender_acc: str = Body(...), 
    receiver_acc: str = Body(...), 
    amount: float = Body(...)
):
    # We use a combined key for the thread_id to keep TX history unique
    config = {"configurable": {"thread_id": f"tx_{sender_acc}"}}
    
    initial_state = {
        "sender_account_number": sender_acc,
        "receiver_account_number": receiver_acc,
        "money": amount,
        "messages": []
    }
    
    final_state = transfer_workflow.invoke(initial_state, config)
    snapshot = transfer_workflow.get_state(config)
    
    if snapshot.next:
        return {"status": "flagged", "message": "Transaction held for fraud review."}
        
    return {
        "status": final_state.get("transaction_status"),
        "message": final_state.get("notification_message")
    }

    