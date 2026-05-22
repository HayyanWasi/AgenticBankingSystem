from fastapi import APIRouter, Body
from app.graphs.transfer_graph import transfer_workflow
import uuid

router = APIRouter()

@router.post("/initiate")
async def initiate_transfer(
    sender_acc: str = Body(...),
    receiver_acc: str = Body(...),
    amount: float = Body(...)
):
    print(f"\n--- [DEBUG] INITIATE TRANSFER API CALLED ---")
    print(f"Sender: {sender_acc} | Receiver: {receiver_acc} | Amount: {amount}")
    print(f"--------------------------------------------\n")

    # Unique thread per transfer — MemorySaver must never replay an old state
    config = {"configurable": {"thread_id": f"tx_{uuid.uuid4()}"}}

    initial_state = {
        "sender_account_number": sender_acc,
        "receiver_account_number": receiver_acc,
        "money": amount,
        "messages": []
    }

    # The graph itself handles the DB transfer inside the process_transfer node.
    # Do NOT call transfer_funds again here — that would double-deduct the money.
    final_state = transfer_workflow.invoke(initial_state, config)
    snapshot = transfer_workflow.get_state(config)

    if snapshot.next:
        return {"status": "flagged", "message": "Transaction held for fraud review."}

    transaction_status = final_state.get("transaction_status")
    notification_message = final_state.get("notification_message", "")
    reason = final_state.get("reason", "")

    # Any non-completed status is an error
    if transaction_status != "completed":
        error_msg = reason or notification_message or f"Transfer could not be completed (status: {transaction_status})."
        return {"status": "failed", "message": error_msg}

    return {
        "status": "completed",
        "message": notification_message or f"Transfer of ${amount} completed successfully."
    }