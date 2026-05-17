from app.schemas.payment_transaction_process_schema import TransferState

from app.database.db_tool import transfer_funds

def process_transfer(state: TransferState):
    # Pull requirements from the state
    sender = state['sender_account_number']
    receiver = state['receiver_account_number']
    amount = state['money']

    # Final execution in the database
    result = transfer_funds(sender, receiver, amount)

    if result["status"] == "success":
        return {"transaction_status": "completed"}
    else:
        return {
            "transaction_status": "failed",
            "reason": result["reason"]
        }