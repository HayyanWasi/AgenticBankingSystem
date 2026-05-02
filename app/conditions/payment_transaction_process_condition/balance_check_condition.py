from app.schemas.payment_transaction_process_schema import TransferState

def balance_check_condition(state: TransferState):
    if state['balance_check_status'] == 'insufficient_balance':
        return 'notify_customer'
    else:
        return 'fraud_check'