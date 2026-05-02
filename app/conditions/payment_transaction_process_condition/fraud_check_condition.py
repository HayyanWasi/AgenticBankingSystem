from app.schemas.payment_transaction_process_schema import TransferState

def fraud_check_condition(state: TransferState):
    if state['is_fraud'] == True:
        return "human_review"
    return "process_transfer"
   