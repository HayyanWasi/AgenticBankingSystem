from app.schemas.payment_transaction_process_schema import TransferState

def human_review_condition(state: TransferState):
    if state['human_decision'] == 'approve':
        return 'process_transfer'
    return 'rejected'
