from app.schemas.payment_transaction_process_schema import TransferState
from app.database.db_tool import get_account_balance

def balance_check(state: TransferState):
    sender_account = state.get('sender_account_number')
    amount_to_send = state.get('money')

    db_data = get_account_balance(sender_account)

    if db_data["status"] == "failed":
        return {
            "balance_check_status": "failed",
            "transaction_status": "failed",
            "reason": db_data["reason"]
        }

    # Compare truth vs. request
    if db_data["balance"] >= amount_to_send:
        return {
            "balance_check_status": "passed",
            "total_balance": db_data["balance"] # Update state with the real balance
        }
    
    return {
        "balance_check_status": "insufficient_balance",
        "transaction_status": "failed",
        "reason": f"Insufficient funds. Your balance is ${db_data['balance']:.2f}."
    }