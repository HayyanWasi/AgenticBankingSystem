from app.schemas.payment_transaction_process_schema import TransferState

def fraud_check_condition(state: TransferState):
    # .get() prevents crashes if the key doesn't exist
    if state.get('is_fraud'): 
        return "human_review"
    
    return "process_transfer"